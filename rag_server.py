import os
import sys
import csv
import json
import re
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
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

from openpyxl import load_workbook

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
STUDENTS_DB_PATH = DATA_DIR / "students.json"
EXCEL_STUDENTS_PATH = DATA_DIR / "students_ctxh.xlsx"

# Tien to tim kiem file nguon du lieu hoat dong CTXH (moi dinh dang)
CTXH_SOURCE_PREFIX = "ctxh_activities"
CTXH_SOURCE_EXTENSIONS = {".xlsx", ".xlsm", ".csv", ".json", ".docx", ".md", ".txt"}

# File JSON luu noi dung quy dinh va cach tinh diem CTXH da duoc boc tach
CTXH_RULES_JSON_PATH = DATA_DIR / "ctxh.json"
CTXH_RULES_DOC_EXTENSIONS = {".md", ".docx", ".txt", ".xlsx", ".pdf"}

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
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        if p.name == "students_ctxh.xlsx" or p.name.startswith("ctxh_activities"):
            continue
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
            try:
                docs = DocumentLoaderFactory.load_file(fpath)
                chunks = chunker.split_documents(docs)
            except Exception as e:
                sys.stderr.write(f"[Warning] Failed to load/chunk {fname}: {e}\n")
                continue

            if is_modified:
                STORE.remove_chunks_by_file(fname)
                updated_files.append(fname)
            else:
                added_files.append(fname)

            if chunks:
                embeddings = None
                if emb_gen is None:
                    try:
                        emb_gen = EmbeddingGenerator()
                    except Exception as e:
                        sys.stderr.write(f"[Warning] EmbeddingGenerator unavailable: {e}\n")
                        emb_gen = None

                if emb_gen is not None:
                    try:
                        texts = [c.content for c in chunks]
                        embeddings = emb_gen.embed_documents(texts, batch_size=16)
                    except Exception as e:
                        sys.stderr.write(f"[Warning] Failed to compute embeddings for {fname}: {e}\n")
                        embeddings = None

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
    """Load cached index, sync students from excel, sync CTXH activities, extract CTXH rules to json, and synchronize data/."""
    global STORE
    if INDEX_PATH.exists():
        try:
            STORE = VectorStore.load_from_json(INDEX_PATH)
        except Exception:
            STORE = VectorStore()

    sync_knowledge_base()
    sync_students_from_excel()
    sync_ctxh_from_file()
    sync_ctxh_rules_from_file()


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


def sync_students_from_excel() -> dict:
    """
    Doc file Excel students_ctxh.xlsx va dong bo toan bo vao students.json.
    Neu khong tim thay file Excel, tu dong xoa sach danh sach sinh vien ve rong [].
    """
    if not EXCEL_STUDENTS_PATH.exists():
        # Xoa sach students.json ve rong []
        STUDENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STUDENTS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return {
            "status": "removed",
            "message": "File students_ctxh.xlsx khong ton tai trong data/. Da xoa sach danh sach sinh vien trong students.json ve rong [].",
        }

    try:
        wb = load_workbook(str(EXCEL_STUDENTS_PATH), data_only=True)
        sheet_name = "students" if "students" in wb.sheetnames else wb.sheetnames[0]
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except Exception as e:
        return {"status": "error", "message": f"Loi khi doc file Excel: {e}"}

    if not rows or len(rows) < 2:
        STUDENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STUDENTS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return {"status": "empty", "message": "File Excel khong co du lieu. Da lam rong students.json."}

    headers = [str(h).strip().lower() if h else f"col_{i}" for i, h in enumerate(rows[0])]
    students = []

    # Doc du lieu CTXH hien co de giu nguyen cac hoat dong da dang ky
    ctxh_records = _load_student_ctxh()
    existing_ctxh_by_mssv = {s.get("mssv"): s for s in ctxh_records}
    updated_ctxh = False

    for r in rows[1:]:
        if not any(r):
            continue
        row_dict = dict(zip(headers, r))

        mssv = str(row_dict.get("mssv", "")).strip()
        if mssv.endswith(".0"):
            mssv = mssv[:-2]
        if not mssv:
            continue

        student_name = str(row_dict.get("student_name", "")).strip()
        faculty = str(row_dict.get("faculty", "")).strip()
        course_year = str(row_dict.get("course_year", "2023")).strip()
        if course_year.endswith(".0"):
            course_year = course_year[:-2]

        program_type = str(row_dict.get("program_type", "")).strip()

        try:
            training_years = float(row_dict.get("training_years") or 4.0)
        except (ValueError, TypeError):
            training_years = 4.0

        try:
            required_ctxh_days = float(row_dict.get("required_ctxh_days") or 4.0)
        except (ValueError, TypeError):
            required_ctxh_days = 4.0

        email = str(row_dict.get("email", "")).strip()
        status = str(row_dict.get("status", "Đang học")).strip()

        student_entry = {
            "mssv": mssv,
            "student_name": student_name,
            "faculty": faculty,
            "course_year": course_year,
            "program_type": program_type,
            "training_years": training_years,
            "required_ctxh_days": required_ctxh_days,
            "email": email,
            "status": status,
        }
        students.append(student_entry)

        # Dong bo ho so sang student_ctxh.json neu chua co hoac cap nhat thong tin co ban
        if mssv not in existing_ctxh_by_mssv:
            existing_ctxh_by_mssv[mssv] = {
                "mssv": mssv,
                "student_name": student_name,
                "faculty": faculty,
                "course_year": course_year,
                "required_days": required_ctxh_days,
                "accumulated_days": 0.0,
                "registered_activities": [],
            }
            updated_ctxh = True
        else:
            rec = existing_ctxh_by_mssv[mssv]
            rec["student_name"] = student_name
            rec["faculty"] = faculty
            rec["course_year"] = course_year
            rec["required_days"] = required_ctxh_days

    STUDENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDENTS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(students, f, ensure_ascii=False, indent=2)

    if updated_ctxh or len(existing_ctxh_by_mssv) != len(ctxh_records):
        _save_student_ctxh(list(existing_ctxh_by_mssv.values()))

    return {
        "status": "success",
        "total_students": len(students),
        "file": str(EXCEL_STUDENTS_PATH.name),
    }


def _find_ctxh_rules_source_file() -> Path | None:
    """Tim kiem file quy che CTXH (nhu CTXH.md, CTXH.docx,...) trong thu muc data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    default_md = DATA_DIR / "CTXH.md"
    if default_md.exists():
        return default_md
    for p in DATA_DIR.glob("*"):
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        if p.name.lower() in {"ctxh.json", "student_ctxh.json", "students.json", "index.json"}:
            continue
        if p.name.lower().startswith("ctxh_activities"):
            continue
        if p.is_file() and p.stem.lower() == "ctxh" and p.suffix.lower() in CTXH_RULES_DOC_EXTENSIONS:
            return p
    return None


def _parse_ctxh_rules_from_file(fpath: Path) -> dict:
    """
    Boc tach noi dung ve cach hoat dong va cach tinh diem cho hoat dong CTXH tu file quy che (nhu CTXH.md).
    Tra ve dict gom:
      - calculation_rules: cong thuc quy doi diem sang ngay CTXH, quy tac tru diem, luu y.
      - activities: danh sach cac hoat dong (ma 6 chu so, linh vuc, ten hoat dong & dinh luong, thang diem, quy doi ngay).
    """
    ext = fpath.suffix.lower()
    activities: list[dict] = []
    calculation_rules = {
        "formula": "Lấy tổng điểm hoạt động CTXH SV đã tích lũy chia cho 10 (số lẻ dưới 5 làm tròn xuống; tròn 5 điểm, quy đổi thành nửa ngày CTXH; trên 5 làm tròn thành 10, quy đổi thành một ngày CTXH).",
        "deduction_rules": [
            "SV đăng ký tham gia hoạt động CTXH nhưng không thực hiện mà không có lý do hoặc lý do không chính đáng bị trừ 0,5 ngày CTXH.",
            "SV không hoàn thành nhiệm vụ, gây ảnh hưởng xấu bị trừ từ 01 đến toàn bộ số ngày đã tích lũy."
        ],
        "exclusions": [
            "Các hoạt động không mang tính chất CTXH như xem biểu diễn văn nghệ, xem thể thao, du lịch, tham quan, sinh hoạt CLB không được tính ngày CTXH.",
            "Khi SV tham gia hoạt động CTXH đã được trả tiền công hoặc trả phí thì không được tính ngày CTXH."
        ]
    }

    if ext in {".md", ".txt"}:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            curr_cat = ""
            for line in lines:
                if "|" not in line:
                    continue
                parts = [c.strip() for c in line.split("|")]
                if parts and parts[0] == "":
                    parts = parts[1:]
                if parts and parts[-1] == "":
                    parts = parts[:-1]
                if len(parts) >= 5:
                    cat, code, desc, points, days = parts[0], parts[1], parts[2], parts[3], parts[4]
                    if code and re.match(r"^\d{6}$", code):
                        if cat:
                            curr_cat = cat
                        activities.append({
                            "activity_code": code,
                            "category": curr_cat,
                            "activity_name": desc,
                            "points": points,
                            "converted_days": days,
                            "formula": "Điểm / 10 = Ngày CTXH"
                        })
        except Exception as e:
            print(f"[sync_ctxh_rules] Loi doc md/txt: {e}", file=sys.stderr)

    elif ext in {".xlsx", ".xlsm"}:
        try:
            wb = load_workbook(str(fpath), data_only=True)
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                curr_cat = ""
                for row in ws.iter_rows(values_only=True):
                    row_strs = [str(c).strip() if c is not None else "" for c in row]
                    for idx, val in enumerate(row_strs):
                        if re.match(r"^\d{6}$", val) and len(row_strs) >= idx + 3:
                            code = val
                            cat = row_strs[0] if idx > 0 and row_strs[0] else curr_cat
                            if cat and cat != code:
                                curr_cat = cat
                            desc = row_strs[idx + 1] if len(row_strs) > idx + 1 else ""
                            points = row_strs[idx + 2] if len(row_strs) > idx + 2 else ""
                            days = row_strs[idx + 3] if len(row_strs) > idx + 3 else ""
                            activities.append({
                                "activity_code": code,
                                "category": curr_cat,
                                "activity_name": desc,
                                "points": points,
                                "converted_days": days,
                                "formula": "Điểm / 10 = Ngày CTXH"
                            })
                            break
            wb.close()
        except Exception as e:
            print(f"[sync_ctxh_rules] Loi doc xlsx: {e}", file=sys.stderr)

    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(str(fpath))
            curr_cat = ""
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    for idx, val in enumerate(cells):
                        if re.match(r"^\d{6}$", val) and len(cells) >= idx + 3:
                            code = val
                            cat = cells[0] if idx > 0 and cells[0] else curr_cat
                            if cat and cat != code:
                                curr_cat = cat
                            desc = cells[idx + 1] if len(cells) > idx + 1 else ""
                            points = cells[idx + 2] if len(cells) > idx + 2 else ""
                            days = cells[idx + 3] if len(cells) > idx + 3 else ""
                            activities.append({
                                "activity_code": code,
                                "category": curr_cat,
                                "activity_name": desc,
                                "points": points,
                                "converted_days": days,
                                "formula": "Điểm / 10 = Ngày CTXH"
                            })
                            break
        except Exception as e:
            print(f"[sync_ctxh_rules] Loi doc docx: {e}", file=sys.stderr)

    return {
        "title": "QUY ĐỊNH MÃ HÓA, THANG ĐIỂM & QUY ĐỔI NGÀY CTXH (HCMUTE)",
        "source_file": fpath.name,
        "calculation_rules": calculation_rules,
        "total_activities": len(activities),
        "activities": activities
    }


def sync_ctxh_rules_from_file() -> dict:
    """
    Doc file quy che CTXH (nhu CTXH.md) trong thu muc data/, boc tach noi dung
    ve cach hoat dong va cach tinh diem cho hoat dong CTXH roi nap vao file ctxh.json.
    Neu khong tim thay file quy che -> tu dong xoa sach ctxh.json ve [] va thong bao.
    """
    source_file = _find_ctxh_rules_source_file()

    if source_file is None:
        CTXH_RULES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CTXH_RULES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        msg = (
            "Khong tim thay file quy che CTXH.* (nhu CTXH.md) trong thu muc data/. "
            "Da xoa sach noi dung quy dinh va cach tinh diem trong ctxh.json ve rong []."
        )
        print(f"[sync_ctxh_rules] {msg}", file=sys.stderr)
        return {"status": "removed", "message": msg}

    parsed_data = _parse_ctxh_rules_from_file(source_file)

    if not parsed_data.get("activities"):
        CTXH_RULES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CTXH_RULES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        msg = f"File '{source_file.name}' khong co hoac khong boc tach duoc bang hoat dong CTXH. Da lam rong ctxh.json."
        print(f"[sync_ctxh_rules] {msg}", file=sys.stderr)
        return {"status": "empty", "message": msg}

    CTXH_RULES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CTXH_RULES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    msg = (
        f"Da boc tach thanh cong {parsed_data['total_activities']} hoat dong va cach tinh diem "
        f"tu file '{source_file.name}' vao {CTXH_RULES_JSON_PATH.name}."
    )
    print(f"[sync_ctxh_rules] {msg}", file=sys.stderr)
    return {
        "status": "success",
        "file": source_file.name,
        "total_activities": parsed_data["total_activities"],
        "target_file": CTXH_RULES_JSON_PATH.name
    }


def _find_ctxh_source_file() -> Path | None:
    """Tim kiem file nguon du lieu hoat dong CTXH trong thu muc data/ theo tien to 'ctxh_activities'."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for p in DATA_DIR.glob(f"{CTXH_SOURCE_PREFIX}*"):
        if p.name.startswith("~$"):
            continue
        if p.is_file() and p.suffix.lower() in CTXH_SOURCE_EXTENSIONS:
            return p
    return None


def _parse_ctxh_rows_from_file(fpath: Path) -> list[dict]:
    """
    Phan tich cac ban ghi hoat dong CTXH tu file nguon.
    Ho tro: .xlsx/.xlsm, .csv, .json, .docx, .md/.txt
    Tra ve list of dict voi cac khoa: mssv, activity_code, activity_name, points, converted_days, status, note
    """
    ext = fpath.suffix.lower()
    rows: list[dict] = []

    if ext in {".xlsx", ".xlsm"}:
        try:
            wb = load_workbook(str(fpath), data_only=True)
            ws = wb.active
            raw_rows = list(ws.iter_rows(values_only=True))
            wb.close()
            if len(raw_rows) < 2:
                return []
            headers = [str(h).strip().lower() if h else f"col_{i}" for i, h in enumerate(raw_rows[0])]
            for r in raw_rows[1:]:
                if not any(r):
                    continue
                rows.append(dict(zip(headers, r)))
        except Exception as e:
            print(f"[sync_ctxh] Loi doc xlsx: {e}", file=sys.stderr)

    elif ext == ".csv":
        try:
            with open(fpath, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k.strip().lower(): v for k, v in row.items()})
        except Exception as e:
            print(f"[sync_ctxh] Loi doc csv: {e}", file=sys.stderr)

    elif ext == ".json":
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = [{k.strip().lower(): v for k, v in r.items()} if isinstance(r, dict) else {} for r in data]
        except Exception as e:
            print(f"[sync_ctxh] Loi doc json: {e}", file=sys.stderr)

    elif ext == ".docx":
        try:
            from docx import Document  # python-docx
            doc = Document(str(fpath))
            headers: list[str] = []
            for table in doc.tables:
                for i, row in enumerate(table.rows):
                    cells = [c.text.strip() for c in row.cells]
                    if i == 0:
                        headers = [c.lower() for c in cells]
                    else:
                        if any(cells):
                            rows.append(dict(zip(headers, cells)))
            # Fallback: parse pipe-delimited lines trong paragraphs
            if not rows:
                lines = [p.text.strip() for p in doc.paragraphs if "|" in p.text]
                if lines:
                    headers = [h.strip().lower() for h in lines[0].split("|") if h.strip()]
                    for line in lines[1:]:
                        if set(line.replace("|", "").replace("-", "").replace(" ", "")) == set():
                            continue  # separator row
                        cells = [c.strip() for c in line.split("|") if c.strip() or True]
                        cells = [c.strip() for c in line.split("|")]
                        # Bo cot trong dau va cuoi neu co
                        if cells and cells[0] == "":
                            cells = cells[1:]
                        if cells and cells[-1] == "":
                            cells = cells[:-1]
                        if any(cells):
                            rows.append(dict(zip(headers, cells)))
        except Exception as e:
            print(f"[sync_ctxh] Loi doc docx: {e}", file=sys.stderr)

    elif ext in {".md", ".txt"}:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Tim bang markdown (dong co dau |)
            table_lines = [l.strip() for l in lines if "|" in l]
            if table_lines:
                headers = [h.strip().lower() for h in table_lines[0].split("|") if h.strip()]
                for line in table_lines[1:]:
                    # Bo qua dong phan cach (----)
                    if set(line.replace("|", "").replace("-", "").replace(" ", "").replace(":", "")) == set():
                        continue
                    cells = [c.strip() for c in line.split("|")]
                    if cells and cells[0] == "":
                        cells = cells[1:]
                    if cells and cells[-1] == "":
                        cells = cells[:-1]
                    if any(cells) and len(cells) == len(headers):
                        rows.append(dict(zip(headers, cells)))
        except Exception as e:
            print(f"[sync_ctxh] Loi doc md/txt: {e}", file=sys.stderr)

    return rows


def sync_ctxh_from_file() -> dict:
    """
    Tim kiem file nguon hoat dong CTXH (bat ky dinh dang nao) trong data/,
    phan tich va nap vao student_ctxh.json.
    Neu khong co file -> xoa sach student_ctxh.json ve [].
    """
    source_file = _find_ctxh_source_file()

    if source_file is None:
        # Xoa sach student_ctxh.json ve rong
        _save_student_ctxh([])
        msg = (
            f"Khong tim thay file '{CTXH_SOURCE_PREFIX}.*' trong thu muc data/. "
            "Da xoa sach du lieu hoat dong CTXH trong student_ctxh.json ve rong []."
        )
        print(f"[sync_ctxh] {msg}", file=sys.stderr)
        return {"status": "removed", "message": msg}

    raw_rows = _parse_ctxh_rows_from_file(source_file)

    if not raw_rows:
        _save_student_ctxh([])
        msg = f"File '{source_file.name}' khong co du lieu. Da lam rong student_ctxh.json."
        print(f"[sync_ctxh] {msg}", file=sys.stderr)
        return {"status": "empty", "message": msg}

    # Nhom cac hoat dong theo mssv
    activities_by_mssv: dict[str, list[dict]] = {}
    for row in raw_rows:
        mssv = str(row.get("mssv", "")).strip()
        if mssv.endswith(".0"):
            mssv = mssv[:-2]
        if not mssv:
            continue
        activities_by_mssv.setdefault(mssv, []).append(row)

    # Lay thong tin sinh vien goc de bo sung ten, khoa, khoa hoc
    students_master = _load_students()
    master_by_mssv = {s.get("mssv"): s for s in students_master}

    new_ctxh_records: list[dict] = []
    total_activities = 0

    for mssv, act_list in activities_by_mssv.items():
        m_info = master_by_mssv.get(mssv)
        registered_activities = []
        acc_days = 0.0

        for row in act_list:
            try:
                points = int(float(str(row.get("points", 5)).strip() or 5))
            except (ValueError, TypeError):
                points = 5
            try:
                converted_days = float(str(row.get("converted_days", points / 10)).strip() or points / 10)
            except (ValueError, TypeError):
                converted_days = round(points / 10, 2)

            activity_code = str(row.get("activity_code", "")).strip()
            activity_name = str(row.get("activity_name", "")).strip() or f"Hoat dong CTXH ma {activity_code}"
            status = str(row.get("status", "CONFIRMED")).strip().upper() or "CONFIRMED"
            note = str(row.get("note", "")).strip()
            reg_date = str(row.get("registration_date", datetime.now().strftime("%Y-%m-%d"))).strip()
            # Neu reg_date la float (Excel date serial), chuyen ve string
            try:
                float(reg_date)
                reg_date = datetime.now().strftime("%Y-%m-%d")
            except ValueError:
                pass

            # Tao reg_id neu khong co
            reg_id = str(row.get("reg_id", "")).strip()
            if not reg_id:
                short_id = uuid.uuid4().hex[:6].upper()
                reg_id = f"REG-CTXH-{mssv}-{short_id}"

            registered_activities.append({
                "reg_id": reg_id,
                "activity_code": activity_code,
                "activity_name": activity_name,
                "points": points,
                "converted_days": converted_days,
                "status": status,
                "registration_date": reg_date,
                "note": note,
            })
            if status == "CONFIRMED":
                acc_days += converted_days

        acc_days = round(acc_days, 2)
        total_activities += len(registered_activities)

        new_ctxh_records.append({
            "mssv": mssv,
            "student_name": m_info.get("student_name", "Unknown") if m_info else "Unknown",
            "faculty": m_info.get("faculty", "") if m_info else "",
            "course_year": m_info.get("course_year", "2022") if m_info else "2022",
            "required_days": m_info.get("required_ctxh_days", 4.0) if m_info else 4.0,
            "accumulated_days": acc_days,
            "registered_activities": registered_activities,
        })

    _save_student_ctxh(new_ctxh_records)
    msg = (
        f"Dong bo thanh cong tu '{source_file.name}': "
        f"{len(new_ctxh_records)} sinh vien, {total_activities} hoat dong CTXH."
    )
    print(f"[sync_ctxh] {msg}", file=sys.stderr)
    return {
        "status": "success",
        "file": source_file.name,
        "total_students": len(new_ctxh_records),
        "total_activities": total_activities,
    }


def _load_students() -> list[dict]:
    if not STUDENTS_DB_PATH.exists():
        return []
    try:
        with open(STUDENTS_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# Initialize and sync store on startup
_ensure_index_loaded()


# ==========================================
# MCP TOOLS (Declared for AI Agents to call)
# ==========================================

@mcp.tool()
def search_ctxh_rules(
    query: str,
    top_k: int = 3,
    mode: str = "hybrid",
    year: Optional[str] = None,
) -> str:
    """Tra cuu van ban quy dinh Cong tac xa hoi (CTXH) bang Hybrid Search (BM25 + Dense Vector).
    Tra ve cac dieu khoan, ma hoat dong (6 chu so), thang diem, cach quy doi ngay CTXH, va dieu kien tot nghiep.
    Bat buoc goi cong cu nay khi tra cuu ma hoat dong, quy dinh ngay CTXH hoac tieu chuan mien giam.

    Args:
        query: Tu khoa, ma hoat dong (001101, 001202...), hoac cau hoi can tra cuu quy dinh CTXH.
        top_k: So luong trich doan can lay (mac dinh 3, toi da 6).
        mode: Che do tim kiem ('hybrid', 'vector', 'bm25').
        year: Nam ban hanh quy dinh cu the (vi du '2023', '2024', '2025'). Neu de trong, he thong tu dong uu tien ban moi nhat.
    """
    top_k = max(1, min(top_k, 6))
    results = STORE.search(query=query, top_k=top_k, mode=mode, year=year)

    if not results:
        return "Khong tim thay quy dinh phu hop trong co so du lieu CTXH."

    # Loc lai lan nua: voi cac chunk cung doc_id, chi giu version lon nhat
    if not year:
        from src.retrieval.vector_store import parse_version_key
        max_v_in_results: dict[str, tuple] = {}
        for chunk, _ in results:
            d_id = chunk.doc_id or (chunk.metadata or {}).get("doc_id", "")
            vk = parse_version_key((chunk.metadata or {}).get("version"))
            if d_id not in max_v_in_results or vk > max_v_in_results[d_id]:
                max_v_in_results[d_id] = vk

        results = [
            (chunk, score) for chunk, score in results
            if parse_version_key((chunk.metadata or {}).get("version")) == max_v_in_results.get(
                chunk.doc_id or (chunk.metadata or {}).get("doc_id", "")
            )
        ]

    formatted_passages = []
    for rank, (chunk, score) in enumerate(results, 1):
        meta = chunk.metadata or {}
        file_name = meta.get("file_name", "unknown")
        version = meta.get("version") or "Chưa xác định"
        page_str = f" | Trang {meta.get('page_number')}" if "page_number" in meta else ""

        passage = (
            f"--- [Trich doan {rank}] ---\n"
            f"[Nguồn: {file_name}{page_str} | Phiên bản: {version} | Chunk ID: {chunk.chunk_id}]\n"
            f"Do tuong dong: {score:.4f}\n"
            f"Noi dung:\n{chunk.content}"
        )
        formatted_passages.append(passage)

    return "\n\n".join(formatted_passages)


# Alias for backward compatibility if needed
@mcp.tool()
def search_regulations(query: str, top_k: int = 4, year: Optional[str] = None) -> str:
    """Tra cuu quy dinh CTXH. Tuong duong voi search_ctxh_rules."""
    return search_ctxh_rules(query=query, top_k=top_k, mode="hybrid", year=year)


@mcp.tool()
def list_indexed_documents() -> str:
    """Liet ke toan bo cac van ban quy dinh ve CTXH hien co trong he thong."""
    if not STORE.chunks:
        return "Co so du lieu tai lieu CTXH hien dang trong."

    file_summary: dict[str, dict] = {}
    for chunk in STORE.chunks.values():
        meta = chunk.metadata or {}
        fname = meta.get("file_name", chunk.doc_id)
        if fname not in file_summary:
            file_summary[fname] = {
                "doc_id": chunk.doc_id,
                "version": meta.get("version", ""),
                "extension": meta.get("extension", ""),
                "metadata_source": meta.get("metadata_source", "none"),
                "chunk_count": 0,
                "total_pages": meta.get("total_pages"),
            }
        file_summary[fname]["chunk_count"] += 1

    lines = ["Danh muc van ban quy che CTXH hien co:"]
    for fname, info in file_summary.items():
        ver_str = f", Phien ban: {info['version']}" if info['version'] else ""
        source_str = f", Nguon metadata: {info['metadata_source']}"
        pages_info = f", So trang: {info['total_pages']}" if info["total_pages"] else ""
        lines.append(
            f"  - [{info['doc_id']}] {fname} (Dinh dang: {info['extension']}{ver_str}{source_str}, Chunks: {info['chunk_count']}{pages_info})"
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
def check_student_exists(mssv: str) -> str:
    """[READ TOOL] Kiem tra thong tin sinh vien co ton tai trong he thong co so du lieu cua truong hay khong.
    Tra ve Ho ten, Khoa, Khoa hoc, He dao tao, va so ngay CTXH tieu chuan de tot nghiep.
    Nen goi cong cu nay de xac dinh danh tinh sinh vien truoc khi dang ky hoat dong."""
    students = _load_students()
    for s in students:
        if s.get("mssv") == mssv:
            return (
                f"[TIM THAY SINH VIEN]\n"
                f"- MSSV: {s.get('mssv')}\n"
                f"- Ho ten: {s.get('student_name')}\n"
                f"- Khoa: {s.get('faculty')}\n"
                f"- Khoa hoc: {s.get('course_year')}\n"
                f"- He dao tao: {s.get('program_type')} ({s.get('training_years')} nam)\n"
                f"- So ngay CTXH yeu cau de tot nghiep: {s.get('required_ctxh_days')} ngay\n"
                f"- Tinh trang hoc tap: {s.get('status', 'Dang hoc')}"
            )
    return f"Loi: Khong tim thay sinh vien co MSSV '{mssv}' trong co so du lieu sinh vien cua truong HCMUTE."


@mcp.tool()
def check_graduation_eligibility(mssv: str) -> str:
    """[SYNTHESIS/CALCULATION TOOL] Tham dinh va tinh toan tien do tich luy ngay CTXH so voi chuan tot nghiep.
    Doi chieu giua ho so sinh vien (students.json) va so ngay da tich luy thuc te (student_ctxh.json).
    Tra ve so ngay da co, so ngay con thieu, va ket luan sinh vien da du dieu kien tot nghiep hay chua."""
    students = _load_students()
    student_master = next((s for s in students if s.get("mssv") == mssv), None)

    ctxh_data = _load_student_ctxh()
    student_ctxh = next((s for s in ctxh_data if s.get("mssv") == mssv), None)

    if not student_master and not student_ctxh:
        return f"Loi: Khong tim thay thong tin sinh vien co MSSV '{mssv}' trong he thong de kiem tra dieu kien tot nghiep."

    name = student_master.get("student_name") if student_master else student_ctxh.get("student_name", "N/A")
    faculty = student_master.get("faculty") if student_master else student_ctxh.get("faculty", "N/A")
    req_days = student_master.get("required_ctxh_days", 4.0) if student_master else student_ctxh.get("required_days", 4.0)
    acc_days = student_ctxh.get("accumulated_days", 0.0) if student_ctxh else 0.0

    activities = student_ctxh.get("registered_activities", []) if student_ctxh else []
    total_points = sum(a.get("points", 0) for a in activities)
    req_points_est = int(req_days * 10)

    diff = round(req_days - acc_days, 2)
    diff_points = max(0, req_points_est - total_points)

    if diff <= 0:
        status_text = "DA DU DIEU KIEN TOT NGHIEP CTXH"
        detail_msg = f"Sinh vien da hoan tat {acc_days}/{req_days} ngay CTXH (Tong diem tich luy: {total_points} diem)."
    else:
        status_text = f"CHUA DU DIEU KIEN TOT NGHIEP (CON THIEU {diff} NGAY)"
        detail_msg = (
            f"Sinh vien hien co {total_points} diem CTXH, quy doi duoc {acc_days}/{req_days} ngay CTXH. "
            f"Can tich luy them it nhat {diff} ngay nua (tuong duong khoang {diff_points} diem hoat dong)."
        )

    return (
        f"[KET QUA THAM DINH DIEU KIEN TOT NGHIEP CTXH]\n"
        f"- MSSV: {mssv} | Ho ten: {name} | Khoa: {faculty}\n"
        f"- Tong diem CTXH da tich luy: {total_points} diem\n"
        f"- So ngay da quy doi thuc te: {acc_days} ngay (Cong thuc: Tong diem / 10 theo quy dinh)\n"
        f"- So ngay tieu chuan can tich luy: {req_days} ngay (uoc tinh khoang {req_points_est} diem)\n"
        f"- Danh gia tien do: {status_text}\n"
        f"- Chi tiet: {detail_msg}"
    )


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
        # Lay thong tin tu danh sach sinh vien goc neu co
        students_master = _load_students()
        m_info = next((m for m in students_master if m.get("mssv") == mssv), None)
        student = {
            "mssv": mssv,
            "student_name": m_info.get("student_name", "Unknown") if m_info else "Unknown",
            "faculty": m_info.get("faculty", "") if m_info else "",
            "course_year": m_info.get("course_year", "2022") if m_info else "2022",
            "required_days": m_info.get("required_ctxh_days", 4.0) if m_info else 4.0,
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
    total_points = sum(a.get("points", 0) for a in activities)

    act_lines = []
    for a in activities:
        act_lines.append(
            f"  - [{a.get('reg_id')}] Ma: {a.get('activity_code')} | "
            f"Ten: {a.get('activity_name')} | "
            f"Diem: {a.get('points', 0)} diem | "
            f"Ngay quy doi: {a.get('converted_days')} ngay | "
            f"Trang thai: {a.get('status')}"
        )

    acts_text = "\n".join(act_lines) if act_lines else "  (Chua co hoat dong nao)"

    return (
        f"[XAC THUC HO SO CTXH THANH CONG]\n"
        f"- MSSV: {student.get('mssv')} | Sinh vien: {student.get('student_name')}\n"
        f"- Khoa: {student.get('faculty', 'N/A')} | Khoa hoc: {student.get('course_year', 'N/A')}\n"
        f"- Tong diem CTXH tich luy: {total_points} diem\n"
        f"- So ngay tich luy hien tai: {acc} ngay (Quy doi tu tong diem chia 10)\n"
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
def sync_ctxh_activities() -> str:
    try:
        result = sync_ctxh_from_file()
    except Exception as e:
        return f"Loi khi dong bo hoat dong CTXH: {e}"

    if result["status"] == "success":
        return (
            f"[CTXH SYNC - THANH CONG] Da dong bo tu file '{result['file']}'.\n"
            f"  - Tong so sinh vien co hoat dong: {result['total_students']}\n"
            f"  - Tong so ban ghi hoat dong: {result['total_activities']}\n"
            f"  - Du lieu da duoc ghi vao student_ctxh.json."
        )
    elif result["status"] == "removed":
        return (
            f"[CTXH SYNC - DA XOA] {result['message']}\n"
            f"  - student_ctxh.json da duoc dat lai ve rong []."
        )
    elif result["status"] == "empty":
        return (
            f"[CTXH SYNC - TRONG] {result['message']}\n"
            f"  - student_ctxh.json da duoc dat lai ve rong []."
        )
    else:
        return f"[CTXH SYNC - LOI] {result.get('message', 'Loi khong xac dinh.')}"


@mcp.tool()
def export_ctxh_rules_json() -> str:
    try:
        result = sync_ctxh_rules_from_file()
    except Exception as e:
        return f"Loi khi boc tach quy che CTXH: {e}"

    if result["status"] == "success":
        return (
            f"[CTXH RULES - THANH CONG] Da boc tach du lieu tu file '{result['file']}' vao '{result['target_file']}'.\n"
            f"  - Tong so hoat dong quy doi: {result['total_activities']}\n"
            f"  - Bao gom: Ma hoat dong (6 chu so), linh vuc, thang diem, quy doi ngay va quy tac tinh diem (chia 10).\n"
            f"  - File ket qua: data/{result['target_file']}"
        )
    elif result["status"] == "removed":
        return (
            f"[CTXH RULES - DA XOA] {result['message']}\n"
            f"  - File ctxh.json da duoc dat lai ve rong []."
        )
    elif result["status"] == "empty":
        return (
            f"[CTXH RULES - TRONG] {result['message']}\n"
            f"  - File ctxh.json da duoc dat lai ve rong []."
        )
    else:
        return f"[CTXH RULES - LOI] {result.get('message', 'Loi khong xac dinh.')}"


@mcp.tool()
def sync_documents() -> str:
    """Dong bo hoa va quet lai thu muc data/ de cap nhat cac file quy dinh CTXH, CSDL sinh vien tu Excel, hoat dong CTXH, va file ctxh.json."""
    report = sync_knowledge_base()
    excel_report = sync_students_from_excel()
    ctxh_report = sync_ctxh_from_file()
    rules_report = sync_ctxh_rules_from_file()

    lines = ["Bao cao dong bo co so tri thuc va du lieu CTXH:"]

    # Tai lieu quy dinh RAG
    if report["added"]:
        lines.append(f"  - Tep quy dinh moi them vao: {', '.join(report['added'])}")
    if report["updated"]:
        lines.append(f"  - Tep quy dinh da cap nhat: {', '.join(report['updated'])}")
    if report["removed"]:
        lines.append(f"  - Tep quy dinh da xoa: {', '.join(report['removed'])}")
    if not (report["added"] or report["updated"] or report["removed"]):
        lines.append("  - Tat ca tai lieu quy dinh da duoc dong bo day du. Khong co thay doi.")
    lines.append(f"  - Tong so van ban: {report['total_files']} tep, {report['total_chunks']} chunks.")

    # CSDL sinh vien tu Excel
    if excel_report.get("status") == "success":
        lines.append(f"  - CSDL sinh vien: Dong bo thanh cong {excel_report.get('total_students')} sinh vien tu {excel_report.get('file')}.")
    elif excel_report.get("status") == "removed":
        lines.append("  - CSDL sinh vien: Khong tim thay tep students_ctxh.xlsx. Da xoa toan bo danh sach sinh vien trong students.json ve rong [].")
    elif excel_report.get("status") == "empty":
        lines.append("  - CSDL sinh vien: Tep students_ctxh.xlsx trong, khong co du lieu. Da xoa students.json ve rong [].")
    elif excel_report.get("status") == "error":
        lines.append(f"  - CSDL sinh vien: {excel_report.get('message')}")

    # Hoat dong CTXH tu file nguon
    if ctxh_report.get("status") == "success":
        lines.append(
            f"  - Hoat dong CTXH (sinh vien): Dong bo thanh cong {ctxh_report.get('total_students')} sinh vien, "
            f"{ctxh_report.get('total_activities')} hoat dong tu '{ctxh_report.get('file')}'."
        )
    elif ctxh_report.get("status") == "removed":
        lines.append(
            f"  - Hoat dong CTXH (sinh vien): Khong tim thay file 'ctxh_activities.*'. "
            "Da xoa sach student_ctxh.json ve rong []."
        )
    elif ctxh_report.get("status") == "empty":
        lines.append(f"  - Hoat dong CTXH (sinh vien): {ctxh_report.get('message')}")
    elif ctxh_report.get("status") == "error":
        lines.append(f"  - Hoat dong CTXH (sinh vien): {ctxh_report.get('message')}")

    # Quy che va cach tinh diem CTXH da boc tach vao ctxh.json
    if rules_report.get("status") == "success":
        lines.append(
            f"  - Bang diem CTXH (ctxh.json): Da boc tach thanh cong {rules_report.get('total_activities')} hoat dong va cach tinh diem tu '{rules_report.get('file')}' vao ctxh.json."
        )
    elif rules_report.get("status") == "removed":
        lines.append("  - Bang diem CTXH (ctxh.json): Khong tim thay file quy che CTXH.*. Da xoa sach ctxh.json ve rong [].")
    elif rules_report.get("status") == "empty":
        lines.append(f"  - Bang diem CTXH (ctxh.json): {rules_report.get('message')}")
    elif rules_report.get("status") == "error":
        lines.append(f"  - Bang diem CTXH (ctxh.json): {rules_report.get('message')}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")

