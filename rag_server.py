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
from src.models import StudentRequest

load_dotenv()

# Initialize MCP Server
mcp = MCPServer("rag-knowledge-server")

DATA_DIR = _HERE / "data"
INDEX_PATH = DATA_DIR / "index.json"
REQUESTS_DB_PATH = DATA_DIR / "student_requests.json"

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

            # Load and chunk this file only
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

    # Sync any new or modified files in data/
    sync_knowledge_base()


# Database helper functions for student requests
def _load_student_requests() -> list[dict]:
    if not REQUESTS_DB_PATH.exists():
        return []
    try:
        with open(REQUESTS_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_student_requests(requests_data: list[dict]) -> None:
    REQUESTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REQUESTS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(requests_data, f, ensure_ascii=False, indent=2)


# Initialize and sync store on startup
_ensure_index_loaded()


# ==========================================
# MCP TOOLS (Declared for AI Agents to call)
# ==========================================

@mcp.tool()
def search_regulations(query: str, top_k: int = 4, mode: str = "hybrid") -> str:
    """Tra cuu van ban quy che hoc vu bang Hybrid Search (BM25 + Dense Vector).
    Tra ve cac doan trich quy che lien quan nhat kem theo nguon file_name, Chunk ID va do tuong dong.
    Bat buoc goi cong cu nay khi can tra cuu quy che, dieu kien hoan thi, rut mon, bao luu hoac le phi."""
    top_k = max(1, min(top_k, 8))
    results = STORE.search(query=query, top_k=top_k, mode=mode)

    if not results:
        return "Khong tim thay quy dinh phu hop trong co so du lieu quy che."

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
def search_documents(query: str, top_k: int = 4) -> str:
    """Tra cuu tai lieu tong hop. Tuong duong voi search_regulations."""
    return search_regulations(query=query, top_k=top_k, mode="hybrid")


@mcp.tool()
def list_indexed_documents() -> str:
    """Liet ke toan bo cac van ban quy che va tai lieu hien co trong he thong.
    Bao gom ma doc_id, ten file, dinh dang va so luong chunks."""
    if not STORE.chunks:
        return "Co so du lieu tai lieu hien dang trong."

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

    lines = ["Danh muc van ban quy che hien co:"]
    for doc_id, info in doc_summary.items():
        pages_info = f", So trang: {info['total_pages']}" if info["total_pages"] else ""
        lines.append(
            f"  - [{doc_id}] {info['file_name']} (Dinh dang: {info['extension']}, Chunks: {info['chunk_count']}{pages_info})"
        )

    return "\n".join(lines)


@mcp.tool()
def get_document_full(doc_id: str) -> str:
    """Trich xuat toan van mot van ban theo ma doc_id.
    Su dung khi can doc toan bo dieu khoan chi tiet cua mot quy che."""
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
    """[WRITE TOOL - SIDE EFFECT] Tao va luu ho so don hoc vu cho sinh vien vao he thong du lieu.
    Cac loai don hop le: HOAN_THI, BAO_LUU, RUT_MON, CAP_BANG_DIEM, CAP_GIAY_XAC_NHAN.
    Bat buoc goi tiep cong cu get_request_status ngay sau khi goi tool nay de verify trang thai."""
    requests_data = _load_student_requests()

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:4].upper()
    req_id = f"REQ-{now_str}-{short_id}"

    if not submission_date:
        submission_date = datetime.now().strftime("%Y-%m-%d")

    req_obj = StudentRequest(
        request_id=req_id,
        mssv=mssv,
        student_name=student_name,
        request_type=request_type.upper(),
        course_name=course_name,
        exam_date=exam_date,
        submission_date=submission_date,
        reason=reason,
        attachment=attachment,
        status="PENDING",
        note="Don da duoc tiep nhan tren he thong va dang cho phe duyet.",
    )

    requests_data.append(req_obj.to_dict())
    _save_student_requests(requests_data)

    return (
        f"Tao ho so thanh cong! Ma ho so: {req_id}. "
        f"Sinh vien: {student_name} (MSSV: {mssv}), Loai don: {request_type}. "
        f"Trang thai: PENDING. Hay goi ngay get_request_status('{req_id}') de kiem tra xac thuc."
    )


@mcp.tool()
def get_request_status(request_id: str) -> str:
    """[VERIFY READ TOOL] Kiem tra va xac thuc trang thai ho so don hoc vu that trong he thong du lieu.
    Tra ve thong tin chi tiet cua don gom: Ma don, MSSV, Sinh vien, Loai don, Trang thai va Ghi chu tham dinh."""
    requests_data = _load_student_requests()
    for req in requests_data:
        if req.get("request_id") == request_id:
            return (
                f"[XAC THUC HO SO THANH CONG]\n"
                f"- Ma don: {req.get('request_id')}\n"
                f"- MSSV: {req.get('mssv')} | Sinh vien: {req.get('student_name')}\n"
                f"- Loai don: {req.get('request_type')}\n"
                f"- Mon hoc: {req.get('course_name', 'N/A')}\n"
                f"- Ngay thi: {req.get('exam_date', 'N/A')} | Ngay nop: {req.get('submission_date', 'N/A')}\n"
                f"- Ly do: {req.get('reason', 'N/A')}\n"
                f"- Minh chung: {req.get('attachment', 'N/A')}\n"
                f"- Trang thai: {req.get('status')}\n"
                f"- Ghi chu: {req.get('note', '')}\n"
                f"- Thoi gian ghi nhan: {req.get('created_at', 'N/A')}"
            )

    return f"Loi xac thuc: Khong tim thay ho so nao voi ma '{request_id}' trong he thong."


@mcp.tool()
def list_student_requests(mssv: str) -> str:
    """Tra cuu toan bo lich su cac don thu tuc hoc vu cua mot sinh vien dua vao MSSV."""
    requests_data = _load_student_requests()
    matched = [r for r in requests_data if r.get("mssv") == mssv]

    if not matched:
        return f"Khong tim thay lich su don nao cho sinh vien co MSSV '{mssv}'."

    lines = [f"Lich su ho so hoc vu cua MSSV {mssv}:"]
    for r in matched:
        lines.append(
            f"  - [{r.get('request_id')}] Loai: {r.get('request_type')} | "
            f"Trang thai: {r.get('status')} | Ngay nop: {r.get('submission_date')} | "
            f"Ghi chu: {r.get('note')}"
        )

    return "\n".join(lines)


@mcp.tool()
def sync_documents() -> str:
    """Dong bo hoa va quet lai thu muc data/ de cap nhat cac file quy che moi hoac sua doi."""
    report = sync_knowledge_base()
    lines = ["Bao cao dong bo co so tri thuc quy che:"]
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
