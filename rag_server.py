import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

# Set UTF-8 encoding for standard streams
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.loaders import DocumentLoaderFactory, SUPPORTED_EXTENSIONS
from src.chunking import RecursiveChunker
from src.retrieval import VectorStore, EmbeddingGenerator
from src.models import StudentCTXHRecord, CTXHRegistration

load_dotenv()

# Initialize MCP Server
mcp = MCPServer("rag-knowledge-server")

DATA_DIR = _HERE / "data"
INDEX_PATH = DATA_DIR / "index.json"
CTXH_DB_PATH = DATA_DIR / "student_ctxh.json"

# Global Store
STORE = VectorStore()


def sync_knowledge_base() -> dict:
    """
    Incremental synchronization of data/ directory:
    - Finds new files -> chunks, embeds, adds to store.
    - Finds modified files -> removes old chunks, re-chunks, re-embeds.
    - Finds deleted files -> removes chunks from store.
    - Saves updated store to index.json.
    """
    global STORE
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover all supported files currently on disk in data/
    current_files = {}
    for p in DATA_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            current_files[p.name] = (p, p.stat().st_mtime)

    # Populate file_meta if missing from existing chunks
    if not STORE.file_meta and STORE.chunks:
        for chunk in STORE.chunks.values():
            fname = chunk.metadata.get("file_name")
            if fname and fname in current_files and fname not in STORE.file_meta:
                STORE.file_meta[fname] = current_files[fname][1]

    added_files = []
    updated_files = []
    removed_files = []

    # 2. Check for deleted files (was in store, but no longer on disk)
    for indexed_file in list(STORE.file_meta.keys()):
        if indexed_file not in current_files:
            STORE.remove_chunks_by_file(indexed_file)
            removed_files.append(indexed_file)

    # 3. Check for new or modified files
    emb_gen = None
    chunker = RecursiveChunker(chunk_size=600, chunk_overlap=80)

    for fname, (fpath, mtime) in current_files.items():
        is_new = fname not in STORE.file_meta
        is_modified = not is_new and mtime > STORE.file_meta[fname]

        if is_new or is_modified:
            if is_modified:
                STORE.remove_chunks_by_file(fname)
                updated_files.append(fname)
            else:
                added_files.append(fname)

            docs = DocumentLoaderFactory.load_file(fpath)
            chunks = chunker.split_documents(docs)

            if chunks:
                if emb_gen is None:
                    emb_gen = EmbeddingGenerator()
                texts = [c.content for c in chunks]
                embeddings = emb_gen.embed_documents(texts, batch_size=16)
                STORE.add_chunks(chunks, embeddings)

            STORE.file_meta[fname] = mtime

    # 4. Save to index.json if any changes occurred
    if added_files or updated_files or removed_files or not INDEX_PATH.exists():
        STORE.save_to_json(INDEX_PATH)

    return {
        "added": added_files,
        "updated": updated_files,
        "removed": removed_files,
        "total_chunks": len(STORE.chunks),
        "total_files": len(STORE.file_meta),
    }


def _ensure_index_loaded() -> None:
    """Load cached index and incrementally synchronize with data/."""
    global STORE
    if INDEX_PATH.exists():
        try:
            STORE = VectorStore.load_from_json(INDEX_PATH)
        except Exception:
            STORE = VectorStore()

    sync_knowledge_base()


# Database helper functions for CTXH records
def _load_student_ctxh() -> list[dict]:
    if not CTXH_DB_PATH.exists():
        return []
    try:
        with open(CTXH_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_student_ctxh(data: list[dict]) -> None:
    CTXH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CTXH_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Initialize and sync store on startup
_ensure_index_loaded()


# ==========================================
# MCP TOOLS (Declared for AI Agents to call)
# ==========================================

@mcp.tool()
def search_ctxh_rules(query: str, top_k: int = 4, mode: str = "hybrid") -> str:
    """Tra cuu van ban quy dinh Cong tac xa hoi (CTXH) bang Hybrid Search (BM25 + Dense Vector).
    Tra ve cac dieu khoan, ma hoat dong (6 chu so), thang diem, cach quy doi ngay CTXH, va dieu kien tot nghiep.
    Bat buoc goi cong cu nay khi tra cuu ma hoat dong, quy dinh ngay CTXH hoac tieu chuan mien giam."""
    top_k = max(1, min(top_k, 8))
    results = STORE.search(query=query, top_k=top_k, mode=mode)

    if not results:
        return "Khong tim thay quy dinh phu hop trong co so du lieu CTXH."

    formatted_passages = []
    for rank, (chunk, score) in enumerate(results, 1):
        meta = chunk.metadata or {}
        file_name = meta.get("file_name", "unknown")
        page_str = f" | Trang {meta.get('page_number')}" if "page_number" in meta else ""

        passage = (
            f"--- [Trich doan {rank}] ---\n"
            f"Nguon: {file_name}{page_str}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Do tuong dong: {score:.4f}\n"
            f"Noi dung:\n{chunk.content}"
        )
        formatted_passages.append(passage)

    return "\n\n".join(formatted_passages)


# Alias for backward compatibility if needed
@mcp.tool()
def search_regulations(query: str, top_k: int = 4) -> str:
    """Tra cuu quy dinh CTXH. Tuong duong voi search_ctxh_rules."""
    return search_ctxh_rules(query=query, top_k=top_k, mode="hybrid")


@mcp.tool()
def list_indexed_documents() -> str:
    """Liet ke toan bo cac van ban quy dinh ve CTXH hien co trong he thong."""
    if not STORE.chunks:
        return "Co so du lieu tai lieu CTXH hien dang trong."

    doc_summary: dict[str, dict] = {}
    for chunk in STORE.chunks.values():
        meta = chunk.metadata or {}
        doc_id = chunk.doc_id
        if doc_id not in doc_summary:
            doc_summary[doc_id] = {
                "file_name": meta.get("file_name", doc_id),
                "extension": meta.get("extension", ""),
                "chunk_count": 0,
                "total_pages": meta.get("total_pages"),
            }
        doc_summary[doc_id]["chunk_count"] += 1

    lines = ["Danh muc van ban quy che CTXH hien co:"]
    for doc_id, info in doc_summary.items():
        pages_info = f", So trang: {info['total_pages']}" if info["total_pages"] else ""
        lines.append(
            f"  - [{doc_id}] {info['file_name']} (Dinh dang: {info['extension']}, Chunks: {info['chunk_count']}{pages_info})"
        )

    return "\n".join(lines)


@mcp.tool()
def get_document_full(doc_id: str) -> str:
    """Trich xuat toan van mot van ban quy dinh CTXH theo ma doc_id."""
    matched_chunks = sorted(
        [c for c in STORE.chunks.values() if c.doc_id == doc_id],
        key=lambda c: c.chunk_index,
    )

    if not matched_chunks:
        return f"Khong tim thay van ban nao voi ma doc_id '{doc_id}'."

    meta = matched_chunks[0].metadata or {}
    file_name = meta.get("file_name", doc_id)

    full_text = "\n\n".join(c.content for c in matched_chunks)
    if len(full_text) > 4000:
        full_text = full_text[:4000] + "\n[ ... van ban da duoc cat ngan de dam bao an toan ngu canh ... ]"

    return f"=== Toan van van ban: {file_name} ({doc_id}) ===\n\n{full_text}"


@mcp.tool()
def register_ctxh_activity(
    mssv: str,
    activity_code: str,
    activity_name: str = "",
    points: int = 5,
    converted_days: float = 0.5,
    note: str = "",
) -> str:
    """[WRITE TOOL - SIDE EFFECT] Dang ky hoat dong CTXH va ghi nhan vao ho so sinh vien trong he thong du lieu.
    Cap nhat so ngay tich luy thuc te cua sinh vien.
    Bat buoc goi tiep cong cu get_ctxh_status ngay sau khi goi tool nay de verify trang thai ho so.
    If the student does not exist, a new record is created with default values.
    """
    data = _load_student_ctxh()
    student = None
    for s in data:
        if s.get("mssv") == mssv:
            student = s
            break
    if not student:
        # Auto-create a new student record with minimal info
        student = {
            "mssv": mssv,
            "student_name": "Unknown",
            "faculty": "",
            "course_year": "2022",
            "required_days": 4.0,
            "accumulated_days": 0.0,
            "registered_activities": [],
        }
        data.append(student)
    now_str = datetime.now().strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:4].upper()
    reg_id = f"REG-CTXH-{now_str}-{short_id}"
    code_names = {
        "001202": "Truc tiep hien mau tinh nguyen (01 lan)",
        "001201": "To chuc hoat dong hien mau tinh nguyen",
        "001101": "Mua he xanh - Suot chien dich tai tinh (01 thang)",
        "001104": "Mua he xanh - Doi hinh chuyen 02 tuan",
        "001105": "Mua he xanh - Doi hinh chuyen 01 tuan",
        "001301": "Xuan tinh nguyen (04 ngay)",
        "001401": "Dem hoi trang ram (04 ngay)",
        "001501": "Tiep suc mua thi - Cap thanh (01 thang)",
        "001502": "Tiep suc mua thi - Cap truong (02 tuan)",
        "002101": "Ngay chu nhat xanh (01 buoi)",
        "002102": "Ngay chu nhat xanh (01 ngay)",
    }
    if not activity_name:
        activity_name = code_names.get(activity_code, f"Hoat dong CTXH ma {activity_code}")
    new_reg = {
        "reg_id": reg_id,
        "activity_code": activity_code,
        "activity_name": activity_name,
        "points": points,
        "converted_days": converted_days,
        "status": "CONFIRMED",
        "registration_date": datetime.now().strftime("%Y-%m-%d"),
        "note": note or "Dang ky thanh cong qua Tro ly CTXH",
    }
    student.setdefault("registered_activities", []).append(new_reg)
    student["accumulated_days"] = round(student.get("accumulated_days", 0.0) + converted_days, 2)
    _save_student_ctxh(data)
    return (
        f"Dang ky hoat dong thanh cong! Ma dang ky: {reg_id}. "
        f"Sinh vien: {student.get('student_name', 'Unknown')} (MSSV: {mssv}). "
        f"Hoat dong: {activity_name} (Ma: {activity_code}) | Diem: {points} | Ngay quy doi: +{converted_days}. "
        f"Tong ngay tich luy moi: {student['accumulated_days']}/{student.get('required_days', 4.0)} ngay. "
        f"Hay goi ngay get_ctxh_status('{mssv}') de kiem tra xac thuc."
    )


# Alias for compatibility with general prompt
@mcp.tool()
def submit_academic_request(
    mssv: str,
    student_name: str,
    request_type: str,
    course_name: str = "",
    exam_date: str = "",
    submission_date: str = "",
    reason: str = "",
    attachment: str = "",
) -> str:
    """[WRITE TOOL - ALIAS] Ho tro dang ky hoat dong CTXH qua interface chung."""
    return register_ctxh_activity(
        mssv=mssv,
        activity_code="001202",
        activity_name=f"Dang ky CTXH: {request_type}",
        points=5,
        converted_days=0.5,
        note=f"{reason} (Minh chung: {attachment})",
    )


@mcp.tool()
def get_ctxh_status(mssv: str) -> str:
    """[VERIFY READ TOOL] Kiem tra va xac thuc ho so CTXH thuc te cua sinh vien tu he thong du lieu.
    Tra ve thong tin: MSSV, Ho ten, Khoa, So ngay tich luy, So ngay quy dinh tot nghiep, Tien do va Danh sach hoat dong da tham gia."""
    data = _load_student_ctxh()
    student = None
    for s in data:
        if s.get("mssv") == mssv:
            student = s
            break

    if not student:
        return f"Loi xac thuc: Khong tim thay ho so sinh vien co MSSV '{mssv}' trong he thong."

    acc = student.get("accumulated_days", 0.0)
    req = student.get("required_days", 4.0)
    diff = round(req - acc, 2)
    progress_status = "DA DU DIEU KIEN TOT NGHIEP" if acc >= req else f"CON THIEU {diff} NGAY DE DU DIEU KIEN TOT NGHIEP"

    activities = student.get("registered_activities", [])
    act_lines = []
    for a in activities:
        act_lines.append(
            f"  - [{a.get('reg_id')}] Ma: {a.get('activity_code')} | "
            f"Ten: {a.get('activity_name')} | Ngay quy doi: {a.get('converted_days')} ngay | "
            f"Trang thai: {a.get('status')}"
        )

    acts_text = "\n".join(act_lines) if act_lines else "  (Chua co hoat dong nao)"

    return (
        f"[XAC THUC HO SO CTXH THANH CONG]\n"
        f"- MSSV: {student.get('mssv')} | Sinh vien: {student.get('student_name')}\n"
        f"- Khoa: {student.get('faculty', 'N/A')} | Khoa hoc: {student.get('course_year', 'N/A')}\n"
        f"- So ngay tich luy hien tai: {acc} ngay\n"
        f"- So ngay toi thieu de tot nghiep: {req} ngay\n"
        f"- Danh gia tien do: {progress_status}\n"
        f"- Danh sach hoat dong da ghi nhan:\n{acts_text}"
    )


# Alias for verify tool
@mcp.tool()
def get_request_status(request_id: str) -> str:
    """[VERIFY READ TOOL - ALIAS] Kiem tra trang thai ho so theo MSSV hoac ma dang ky."""
    # If passed MSSV directly
    data = _load_student_ctxh()
    for s in data:
        if s.get("mssv") == request_id:
            return get_ctxh_status(request_id)
        for act in s.get("registered_activities", []):
            if act.get("reg_id") == request_id:
                return get_ctxh_status(s.get("mssv"))
    return f"Loi xac thuc: Khong tim thay ma ho so hoac sinh vien voi ma '{request_id}'."


@mcp.tool()
def sync_documents() -> str:
    """Dong bo hoa va quet lai thu muc data/ de cap nhat cac file quy dinh CTXH moi."""
    report = sync_knowledge_base()
    lines = ["Bao cao dong bo co so tri thuc CTXH:"]
    if report["added"]:
        lines.append(f"  - Tep moi them vao: {', '.join(report['added'])}")
    if report["updated"]:
        lines.append(f"  - Tep da cap nhat: {', '.join(report['updated'])}")
    if report["removed"]:
        lines.append(f"  - Tep da xoa: {', '.join(report['removed'])}")
    if not (report["added"] or report["updated"] or report["removed"]):
        lines.append("  - Tat ca tai lieu da duoc dong bo day du. Khong co thay doi.")
    lines.append(f"  - Tong so: {report['total_files']} tep, {report['total_chunks']} chunks.")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
