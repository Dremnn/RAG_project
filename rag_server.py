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
STUDENTS_DB_PATH = DATA_DIR / "students.json"

# Ho tro ca student_activities.json va student_ctxh.json
STUDENT_ACTIVITIES_PATHS = [
    DATA_DIR / "student_activities.json",
]

# Ho tro ca students.xlsx va students_ctxh.xlsx
EXCEL_STUDENTS_PATHS = [
    DATA_DIR / "students.xlsx",
]

# Ho tro ca activities.json va ctxh.json
ACTIVITIES_RULES_PATHS = [
    DATA_DIR / "activities.json",
]
CTXH_RULES_JSON_PATH = DATA_DIR / "activities.json"
CTXH_RULES_DOC_EXTENSIONS = {".xlsx", ".xlsm", ".docx", ".doc", ".pdf", ".txt", ".md"}

# Cac file Excel luu hoat dong sinh vien da tham gia
CTXH_STUDENT_ACTIVITY_FILENAMES = [
    "student_activities.xlsx",
]

# Global Store
STORE = VectorStore()


def _get_student_activities_path() -> Path:
    for p in STUDENT_ACTIVITIES_PATHS:
        if p.exists():
            return p
    return STUDENT_ACTIVITIES_PATHS[0]


def _get_excel_students_path() -> Path | None:
    for p in EXCEL_STUDENTS_PATHS:
        if p.exists():
            return p
    return None


def _get_activities_rules_path() -> Path:
    for p in ACTIVITIES_RULES_PATHS:
        if p.exists():
            return p
    return ACTIVITIES_RULES_PATHS[0]


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
        ignored_excel = {name.lower() for name in CTXH_STUDENT_ACTIVITY_FILENAMES} | {"students_ctxh.xlsx", "students.xlsx"}
        if p.name.lower() in ignored_excel:
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
    """Load cached index, sync students from excel, extract CTXH rules to json, and synchronize data/."""
    global STORE
    if INDEX_PATH.exists():
        try:
            STORE = VectorStore.load_from_json(INDEX_PATH)
        except Exception:
            STORE = VectorStore()

    sync_knowledge_base()
    sync_students_from_excel()
    sync_ctxh_rules_from_file()
    sync_ctxh_from_file()


# Database helper functions for CTXH records
def _load_student_ctxh() -> list[dict]:
    path = _get_student_activities_path()
    if not path.exists():
        for alt in STUDENT_ACTIVITIES_PATHS:
            if alt.exists():
                path = alt
                break
        else:
            return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_student_ctxh(data: list[dict]) -> None:
    if not data:
        # Safety guard: never wipe file with empty list if the files already exist and have data
        current_data = _load_student_ctxh()
        if current_data:
            sys.stderr.write("[Warning] Tu choi ghi de danh sach sinh vien rong [] len CSDL da co du lieu.\n")
            return
    for p in STUDENT_ACTIVITIES_PATHS:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def sync_students_from_excel() -> dict:
    """
    Doc file Excel (students.xlsx hoac students_ctxh.xlsx) va dong bo vao students.json.
    DONG THOI bao toan 100% lich su hoat dong da co trong student_activities.json.
    TUYET DOI KHONG reset hoat dong cua sinh vien khi chay lai client.
    """
    excel_path = _get_excel_students_path()
    if excel_path is None:
        if STUDENTS_DB_PATH.exists():
            msg = "Khong tim thay file Excel sinh vien, bao toan 100% du lieu students.json hien co."
            print(f"[sync_students] {msg}", file=sys.stderr)
            return {"status": "kept", "message": msg}
        STUDENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STUDENTS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        msg = "Khong tim thay file Excel sinh vien (students.xlsx) trong data/."
        print(f"[sync_students] {msg}", file=sys.stderr)
        return {
            "status": "removed",
            "message": msg,
        }

    try:
        wb = load_workbook(str(excel_path), data_only=True)
        sheet_name = "students" if "students" in wb.sheetnames else wb.sheetnames[0]
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except Exception as e:
        msg = f"Loi khi doc file Excel: {e}"
        print(f"[sync_students] {msg}", file=sys.stderr)
        return {"status": "error", "message": msg}

    if not rows or len(rows) < 2:
        msg = "File Excel khong co du lieu."
        print(f"[sync_students] {msg}", file=sys.stderr)
        return {"status": "empty", "message": msg}

    headers = [str(h).strip().lower() if h else f"col_{i}" for i, h in enumerate(rows[0])]
    students = []

    # Doc du lieu hoat dong hien co de BAO TOAN 100% LICH SU
    existing_activities_records = _load_student_ctxh()
    existing_by_mssv = {s.get("mssv"): s for s in existing_activities_records}

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

        # Neu sinh vien da co ho so -> BAO TOAN 100% LICH SU HOAT DONG, chi cap nhat thong tin co ban!
        if mssv in existing_by_mssv:
            rec = existing_by_mssv[mssv]
            rec["student_name"] = student_name
            rec["faculty"] = faculty
            rec["course_year"] = course_year
            rec["required_days"] = required_ctxh_days
            # registered_activities va accumulated_days DUOC GIU NGUYEN HOAN TOAN
        else:
            existing_by_mssv[mssv] = {
                "mssv": mssv,
                "student_name": student_name,
                "faculty": faculty,
                "course_year": course_year,
                "required_days": required_ctxh_days,
                "accumulated_days": 0.0,
                "registered_activities": [],
            }

    STUDENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDENTS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(students, f, ensure_ascii=False, indent=2)

    _save_student_ctxh(list(existing_by_mssv.values()))

    msg = f"Dong bo thanh cong {len(students)} sinh vien tu '{excel_path.name}' vao students.json."
    print(f"[sync_students] {msg}", file=sys.stderr)

    return {
        "status": "success",
        "total_students": len(students),
        "file": str(excel_path.name),
    }


def _extract_rules_from_text(text: str, rules_dict: dict) -> None:
    """Boc tach quy tac tinh diem, tru diem va loai tru tu van ban quy che."""
    t_lower = text.lower()
    if "chia cho 10" in t_lower:
        rules_dict["formula"] = "Lấy tổng điểm hoạt động CTXH SV đã tích lũy chia cho 10 (số lẻ dưới 5 làm tròn xuống; tròn 5 điểm, quy đổi thành nửa ngày CTXH; trên 5 làm tròn thành 10, quy đổi thành một ngày CTXH)."

    for line in text.splitlines():
        l_lower = line.lower()
        if ("trừ" in l_lower or "tru " in l_lower) and ("ngày" in l_lower or "ngay" in l_lower or "đrl" in l_lower or "kỷ luật" in l_lower):
            clean = line.strip().lstrip("-*•✓ ").strip()
            if len(clean) > 15 and clean not in rules_dict.setdefault("deduction_rules", []):
                rules_dict["deduction_rules"].append(clean)
        if ("không được tính" in l_lower or "không tính" in l_lower or "thù lao" in l_lower or "tiền công" in l_lower):
            clean = line.strip().lstrip("-*•✓ ").strip()
            if len(clean) > 15 and clean not in rules_dict.setdefault("exclusions", []):
                rules_dict["exclusions"].append(clean)


def _find_all_ctxh_rules_files() -> list[Path]:
    """
    Tim tat ca cac file quy che va bang diem CTXH (.xlsx, .xlsm, .docx, .doc, .pdf, .txt, .md) trong thu muc data/.
    Bo hoan toan viec boc tach tu file .json ma thay bang Excel, Word, PDF va Text.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    # Danh sach cac file Excel sinh vien / hoat dong sinh vien can loai tru khoi tap van ban quy che
    student_excel_names = {
        name.lower() for name in CTXH_STUDENT_ACTIVITY_FILENAMES
    } | {"students.xlsx", "students_ctxh.xlsx", "danh_sach_sinh_vien.xlsx"}

    # Quet cac file quy che trong data/ (tuyet doi khong doc file .json)
    for p in sorted(DATA_DIR.glob("*")):
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        # Bo qua tat ca cac file JSON (khong dung json de boc tach nua)
        if p.suffix.lower() == ".json":
            continue
        # Bo qua file Excel danh sach sinh vien va hoat dong da tham gia
        if p.name.lower() in student_excel_names:
            continue
        if p.is_file() and p.suffix.lower() in CTXH_RULES_DOC_EXTENSIONS:
            if p not in files:
                files.append(p)

    def _file_sort_key(p: Path) -> tuple[int, str]:
        m = re.search(r"20\d\d", p.name)
        yr = int(m.group(0)) if m else 0
        return (yr, p.name)

    files.sort(key=_file_sort_key, reverse=True)
    return files



def _find_ctxh_rules_source_file() -> Path | None:
    """Tim kiem mot file quy che CTXH uu tien trong data/."""
    files = _find_all_ctxh_rules_files()
    return files[0] if files else None


def _parse_ctxh_rules_from_file(fpath: Path) -> dict:
    """
    Boc tach noi dung ve danh muc hoat dong va cach tinh diem CTXH tu cac file doc, docx, pdf, json, txt.
    Ho tro da dinh dang:
      - .json: doc danh muc hoat dong va calculation_rules truc tiep.
      - .docx / .doc: doc bang bieu va cac doan van quy dinh.
      - .pdf: trich xuat text va cac bang bieu ma 6 chu so qua pypdf.
      - .txt / .md: doc bang markdown (|) va noi dung quy che.
    Tra ve dict gom calculation_rules va danh sach activities.
    """
    ext = fpath.suffix.lower()
    activities: list[dict] = []
    calculation_rules = {
        "formula": "Lấy tổng điểm hoạt động CTXH SV đã tích lũy chia cho 10 (số lẻ dưới 5 làm tròn xuống; tròn 5 điểm, quy đổi thành nửa ngày CTXH; trên 5 làm tròn thành 10, quy đổi thành một ngày CTXH).",
        "deduction_rules": [],
        "exclusions": []
    }


    if ext in {".docx", ".doc"}:
        try:
            from docx import Document
            doc = Document(str(fpath))
            curr_cat = ""
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
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
            full_doc_text = "\n".join(p.text for p in doc.paragraphs)
            _extract_rules_from_text(full_doc_text, calculation_rules)
        except Exception as e:
            sys.stderr.write(f"[sync_ctxh_rules] Loi doc docx {fpath.name}: {e}\n")

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(fpath))
            full_pdf_text = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                full_pdf_text.append(txt)
            all_text = "\n".join(full_pdf_text)

            curr_cat = ""
            for line in all_text.splitlines():
                line_str = line.strip()
                if not line_str:
                    continue
                m = re.search(r"\b(\d{6})\b", line_str)
                if m:
                    code = m.group(1)
                    if "|" in line_str:
                        parts = [c.strip() for c in line_str.split("|")]
                        if parts and parts[0] == "":
                            parts = parts[1:]
                        if parts and parts[-1] == "":
                            parts = parts[:-1]
                        for idx, p in enumerate(parts):
                            if p == code and len(parts) >= idx + 3:
                                cat = parts[0] if idx > 0 and parts[0] else curr_cat
                                if cat and cat != code:
                                    curr_cat = cat
                                desc = parts[idx + 1]
                                points = parts[idx + 2]
                                days = parts[idx + 3] if len(parts) > idx + 3 else ""
                                activities.append({
                                    "activity_code": code,
                                    "category": curr_cat,
                                    "activity_name": desc,
                                    "points": points,
                                    "converted_days": days,
                                    "formula": "Điểm / 10 = Ngày CTXH"
                                })
                                break
                    else:
                        parts = line_str.split(code, 1)
                        prefix = parts[0].strip()
                        suffix = parts[1].strip()
                        if prefix and not any(ch.isdigit() for ch in prefix):
                            curr_cat = prefix
                        score_matches = list(re.finditer(r"(\d+(?:[.,]\d+)?(?:\s*[→\->]\s*\d+(?:[.,]\d+)?)?)", suffix))
                        if len(score_matches) >= 2:
                            points_match = score_matches[-2].group(1).strip()
                            days_match = score_matches[-1].group(1).strip()
                            desc_end = score_matches[-2].start()
                            desc = suffix[:desc_end].strip().strip("-:| ")
                            activities.append({
                                "activity_code": code,
                                "category": curr_cat,
                                "activity_name": desc or f"Hoạt động CTXH {code}",
                                "points": points_match,
                                "converted_days": days_match,
                                "formula": "Điểm / 10 = Ngày CTXH"
                            })
            _extract_rules_from_text(all_text, calculation_rules)
        except Exception as e:
            sys.stderr.write(f"[sync_ctxh_rules] Loi doc pdf {fpath.name}: {e}\n")

    elif ext in {".md", ".txt"}:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            curr_cat = ""
            for line in content.splitlines():
                if "|" not in line:
                    continue
                parts = [c.strip() for c in line.split("|")]
                code_idx = -1
                for idx, p in enumerate(parts):
                    if re.match(r"^\d{6}$", p):
                        code_idx = idx
                        break
                if code_idx != -1:
                    code = parts[code_idx]
                    cat_candidates = [parts[i] for i in range(code_idx) if parts[i]]
                    if cat_candidates:
                        curr_cat = cat_candidates[-1]
                    desc = parts[code_idx + 1] if len(parts) > code_idx + 1 else ""
                    points = parts[code_idx + 2] if len(parts) > code_idx + 2 else ""
                    days = parts[code_idx + 3] if len(parts) > code_idx + 3 else ""
                    activities.append({
                        "activity_code": code,
                        "category": curr_cat,
                        "activity_name": desc,
                        "points": points,
                        "converted_days": days,
                        "formula": "Điểm / 10 = Ngày CTXH"
                    })
            _extract_rules_from_text(content, calculation_rules)
        except Exception as e:
            sys.stderr.write(f"[sync_ctxh_rules] Loi doc md/txt {fpath.name}: {e}\n")

    elif ext in {".xlsx", ".xlsm"}:
        try:
            wb = load_workbook(str(fpath), data_only=True)
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                curr_cat = ""
                for row in ws.iter_rows(values_only=True):
                    row_strs = [str(c).strip() if c is not None else "" for c in row]
                    code_idx = -1
                    for idx, val in enumerate(row_strs):
                        if re.match(r"^\d{6}$", val):
                            code_idx = idx
                            break
                    if code_idx != -1:
                        code = row_strs[code_idx]
                        cat_candidates = [row_strs[i] for i in range(code_idx) if row_strs[i]]
                        if cat_candidates:
                            curr_cat = cat_candidates[-1]
                        desc = row_strs[code_idx + 1] if len(row_strs) > code_idx + 1 else ""
                        points = row_strs[code_idx + 2] if len(row_strs) > code_idx + 2 else ""
                        days = row_strs[code_idx + 3] if len(row_strs) > code_idx + 3 else ""
                        activities.append({
                            "activity_code": code,
                            "category": curr_cat,
                            "activity_name": desc,
                            "points": points,
                            "converted_days": days,
                            "formula": "Điểm / 10 = Ngày CTXH"
                        })
                    else:
                        text_line = " ".join(r for r in row_strs if r)
                        if len(text_line) > 15:
                            _extract_rules_from_text(text_line, calculation_rules)
            wb.close()
        except Exception as e:
            sys.stderr.write(f"[sync_ctxh_rules] Loi doc xlsx {fpath.name}: {e}\n")

    return {
        "title": "QUY ĐỊNH MÃ HÓA, THANG ĐIỂM & QUY ĐỔI NGÀY CTXH (HCMUTE)",
        "source_file": fpath.name,
        "calculation_rules": calculation_rules,
        "total_activities": len(activities),
        "activities": activities
    }


def sync_ctxh_rules_from_file() -> dict:
    """
    Boc tach du lieu hoat dong CTXH va cach tinh diem tu cac file Excel (.xlsx), Word (.docx), PDF, Text (.txt) trong data/.
    Bo hoan toan viec boc tach tu cac file JSON ma thay bang Excel va cac van ban quy che.
    Ghi ket qua ra data/activities.json va data/ctxh.json.
    """
    rule_files = _find_all_ctxh_rules_files()

    merged_activities: dict[str, dict] = {}
    merged_rules = {
        "formula": "Lấy tổng điểm hoạt động CTXH SV đã tích lũy chia cho 10 (số lẻ dưới 5 làm tròn xuống; tròn 5 điểm, quy đổi thành nửa ngày CTXH; trên 5 làm tròn thành 10, quy đổi thành một ngày CTXH).",
        "deduction_rules": [
            "SV đăng ký tham gia hoạt động CTXH nhưng không thực hiện mà không có lý do chính đáng bị trừ 0,5 ngày CTXH vào tổng số ngày tích lũy.",
            "SV không hoàn thành nhiệm vụ, gây ảnh hưởng xấu bị trừ từ 01 đến toàn bộ số ngày đã tích lũy."
        ],
        "exclusions": [
            "Các hoạt động không mang tính chất CTXH như xem biểu diễn văn nghệ, xem thể thao, du lịch, tham quan, sinh hoạt CLB không được tính ngày CTXH.",
            "Khi SV tham gia hoạt động CTXH đã được trả tiền công hoặc trả phí thì không được tính ngày CTXH."
        ]
    }

    scanned_files: list[str] = []
    for fpath in rule_files:
        try:
            parsed = _parse_ctxh_rules_from_file(fpath)
            scanned_files.append(fpath.name)
            year_match = re.search(r"20\d\d", fpath.name)
            file_year = year_match.group(0) if year_match else ""

            for act in parsed.get("activities", []):
                code = str(act.get("activity_code", "")).strip()
                if not code:
                    continue
                if file_year and not act.get("year"):
                    act["year"] = file_year
                if not act.get("source_file"):
                    act["source_file"] = fpath.name

                if code not in merged_activities:
                    merged_activities[code] = act
                else:
                    for k, v in act.items():
                        if v and not merged_activities[code].get(k):
                            merged_activities[code][k] = v

            calc = parsed.get("calculation_rules", {})
            for r in calc.get("deduction_rules", []):
                if r not in merged_rules["deduction_rules"]:
                    merged_rules["deduction_rules"].append(r)
            for e in calc.get("exclusions", []):
                if e not in merged_rules["exclusions"]:
                    merged_rules["exclusions"].append(e)
            if calc.get("formula") and not merged_rules.get("formula"):
                merged_rules["formula"] = calc["formula"]
        except Exception as e:
            sys.stderr.write(f"[sync_ctxh_rules] Loi khi boc tach file {fpath.name}: {e}\n")

    activities_list = sorted(list(merged_activities.values()), key=lambda x: str(x.get("activity_code", "")))

    if not activities_list:
        msg = "Khong tim thay hoac khong boc tach duoc hoat dong CTXH nao tu data/."
        sys.stderr.write(f"[sync_ctxh_rules] {msg}\n")
        return {"status": "empty", "message": msg, "total_activities": 0}

    final_data = {
        "title": "QUY ĐỊNH MÃ HÓA, THANG ĐIỂM & QUY ĐỔI NGÀY CTXH (HCMUTE)",
        "source_files": scanned_files,
        "calculation_rules": merged_rules,
        "total_activities": len(activities_list),
        "activities": activities_list
    }

    # Luu vao ca activities.json va ctxh.json de dong nhat
    for p in ACTIVITIES_RULES_PATHS:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

    msg = (
        f"Da boc tach thanh cong {len(activities_list)} hoat dong va cach tinh diem "
        f"tu cac file doc/pdf/txt vao activities.json."
    )
    sys.stderr.write(f"[sync_ctxh_rules] {msg}\n")
    return {
        "status": "success",
        "file": ", ".join(scanned_files[:5]),
        "total_activities": len(activities_list),
        "target_file": "activities.json"
    }


def _find_ctxh_source_file() -> Path | None:
    """Tim kiem file Excel luu hoat dong da tham gia CTXH cua sinh vien trong thu muc data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Kiem tra cac ten file Excel uu tien
    for fname in CTXH_STUDENT_ACTIVITY_FILENAMES:
        p = DATA_DIR / fname
        if p.exists() and p.is_file():
            return p

    # 2. Quet cac file Excel khac trong data/ co chua tu khoa hoat dong sinh vien
    for p in sorted(DATA_DIR.glob("*.xlsx")):
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        p_name = p.name.lower()
        if p_name in {"students.xlsx", "students_ctxh.xlsx", "activities.xlsx"}:
            continue
        if any(kw in p_name for kw in ["activity", "activities", "hoat_dong", "tham_gia", "sinh_vien"]):
            return p

    for p in sorted(DATA_DIR.glob("*.xlsm")):
        if p.name.startswith("~$") or p.name.startswith("."):
            continue
        p_name = p.name.lower()
        if any(kw in p_name for kw in ["activity", "activities", "hoat_dong", "tham_gia", "sinh_vien"]):
            return p

    return None


def _normalize_ctxh_record_keys(raw_row: dict) -> dict:
    """Chuan hoa ten cot tu tieng Viet hoac tieng Anh ve khoa chuan cua he thong."""
    normalized = {}
    for k, v in raw_row.items():
        if v is None:
            continue
        key_clean = str(k).strip().lower().replace(" ", "_")
        val = str(v).strip()
        if not val or val.lower() == "none":
            continue

        # MSSV
        if any(w in key_clean for w in ["mssv", "ma_sv", "masv", "ma_sinh_vien", "student_id", "msv"]):
            if val.endswith(".0"):
                val = val[:-2]
            normalized["mssv"] = val
        # Ma hoat dong
        elif any(w in key_clean for w in ["ma_hoat_dong", "ma_hd", "mahd", "activity_code", "ma_so"]):
            if val.endswith(".0"):
                val = val[:-2]
            normalized["activity_code"] = val
        # Ten hoat dong
        elif any(w in key_clean for w in ["ten_hoat_dong", "ten_hd", "tenhd", "hoat_dong", "activity_name", "activity", "noi_dung"]):
            normalized["activity_name"] = val
        # Diem
        elif any(w in key_clean for w in ["diem", "points", "score"]):
            normalized["points"] = val
        # Ngay quy doi
        elif any(w in key_clean for w in ["ngay_quy_doi", "so_ngay", "ngay_ctxh", "converted_days", "days"]):
            normalized["converted_days"] = val
        # Trang thai
        elif any(w in key_clean for w in ["trang_thai", "tinh_trang", "status"]):
            normalized["status"] = val
        # Ngay dang ky / tham gia
        elif any(w in key_clean for w in ["ngay_tham_gia", "ngay_dk", "ngay_dang_ky", "registration_date", "date", "thoi_gian"]):
            normalized["registration_date"] = val
        # Ma dang ky
        elif any(w in key_clean for w in ["ma_dk", "reg_id", "registration_id"]):
            normalized["reg_id"] = val
        # Ghi chu
        elif any(w in key_clean for w in ["ghi_chu", "note", "notes", "nhan_xet"]):
            normalized["note"] = val
        else:
            normalized[key_clean] = val

    # Fallback tu dong nhan dien cot theo gia tri neu header khong khop
    if "mssv" not in normalized:
        for v in raw_row.values():
            s = str(v).strip()
            if s.endswith(".0"):
                s = s[:-2]
            if re.match(r"^\d{7,10}$", s):
                normalized["mssv"] = s
                break
    if "activity_code" not in normalized:
        for v in raw_row.values():
            s = str(v).strip()
            if s.endswith(".0"):
                s = s[:-2]
            if re.match(r"^\d{6}$", s):
                normalized["activity_code"] = s
                break

    return normalized


def _parse_ctxh_rows_from_file(fpath: Path) -> list[dict]:
    """
    Phan tich cac ban ghi hoat dong CTXH da tham gia tu file Excel (.xlsx/.xlsm) hoac .csv.
    Tra ve danh sach cac dict da chuan hoa khoa: mssv, activity_code, activity_name, points, converted_days, status, note, registration_date.
    """
    ext = fpath.suffix.lower()
    rows: list[dict] = []

    if ext in {".xlsx", ".xlsm"}:
        try:
            wb = load_workbook(str(fpath), data_only=True)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                raw_rows = list(ws.iter_rows(values_only=True))
                if len(raw_rows) < 2:
                    continue
                # Tim header row (trong toi da 10 dong dau)
                header_idx = -1
                for r_idx, r in enumerate(raw_rows[:10]):
                    r_strs = [str(c).strip().lower() for c in r if c is not None]
                    if any(any(k in s for k in ["mssv", "sinh_vien", "ma_sv", "hoat_dong", "activity", "code"]) for s in r_strs):
                        header_idx = r_idx
                        break

                if header_idx != -1:
                    raw_headers = [str(h).strip().lower() if h is not None else f"col_{i}" for i, h in enumerate(raw_rows[header_idx])]
                    data_rows = raw_rows[header_idx + 1:]
                else:
                    raw_headers = [str(h).strip().lower() if h is not None else f"col_{i}" for i, h in enumerate(raw_rows[0])]
                    data_rows = raw_rows[1:]

                for r in data_rows:
                    if not any(r):
                        continue
                    r_dict = dict(zip(raw_headers, r))
                    norm = _normalize_ctxh_record_keys(r_dict)
                    if norm.get("mssv") and (norm.get("activity_code") or norm.get("activity_name")):
                        rows.append(norm)
            wb.close()
        except Exception as e:
            sys.stderr.write(f"[sync_ctxh] Loi doc xlsx {fpath.name}: {e}\n")

    elif ext == ".csv":
        try:
            with open(fpath, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    norm = _normalize_ctxh_record_keys(r)
                    if norm.get("mssv") and (norm.get("activity_code") or norm.get("activity_name")):
                        rows.append(norm)
        except Exception as e:
            sys.stderr.write(f"[sync_ctxh] Loi doc csv {fpath.name}: {e}\n")

    return rows


def sync_ctxh_from_file() -> dict:
    """
    Tim kiem file Excel luu hoat dong da tham gia CTXH cua sinh vien trong data/.
    Neu co file: Phan tich va nap bo sung vao student_activities.json.
    Neu khong co file: Tra ve status="none" va TUYET DOI KHONG THAY DOI data/student_activities.json.
    TUYET DOI BAO TOAN LICH SU HOAT DONG, khong bao gio reset ve rong.
    """
    source_file = _find_ctxh_source_file()

    if source_file is None:
        return {
            "status": "none",
            "message": "Khong co file Excel hoat dong da tham gia cua sinh vien. Lich su duoc bao toan nguyen ven.",
        }

    raw_rows = _parse_ctxh_rows_from_file(source_file)

    if not raw_rows:
        msg = f"File '{source_file.name}' khong co du lieu hoat dong moi. Lich su duoc giu nguyen."
        sys.stderr.write(f"[sync_ctxh] {msg}\n")
        return {"status": "kept", "message": msg}

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

    # Tra cuu catalog hoat dong de dien tu dong ten hoat dong / diem neu thieu
    rules_catalog: dict[str, dict] = {}
    for p in ACTIVITIES_RULES_PATHS:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    for act in cdata.get("activities", []):
                        acode = str(act.get("activity_code", "")).strip()
                        if acode:
                            rules_catalog[acode] = act
            except Exception:
                pass

    # Load ho so hien co de MERGE va BAO TOAN 100% LICH SU
    existing_records = _load_student_ctxh()
    records_by_mssv = {r.get("mssv"): r for r in existing_records}

    total_new_activities = 0

    for mssv, act_list in activities_by_mssv.items():
        m_info = master_by_mssv.get(mssv)
        if mssv not in records_by_mssv:
            records_by_mssv[mssv] = {
                "mssv": mssv,
                "student_name": m_info.get("student_name", "Unknown") if m_info else "Unknown",
                "faculty": m_info.get("faculty", "") if m_info else "",
                "course_year": m_info.get("course_year", "2023") if m_info else "2023",
                "required_days": m_info.get("required_ctxh_days", 4.0) if m_info else 4.0,
                "accumulated_days": 0.0,
                "registered_activities": [],
            }
        rec = records_by_mssv[mssv]
        existing_reg_keys = {
            (a.get("activity_code"), a.get("registration_date"))
            for a in rec.get("registered_activities", [])
        }
        existing_reg_ids = {a.get("reg_id") for a in rec.get("registered_activities", []) if a.get("reg_id")}

        for row in act_list:
            activity_code = str(row.get("activity_code", "")).strip()
            catalog_item = rules_catalog.get(activity_code, {})

            activity_name = str(row.get("activity_name", "")).strip()
            if not activity_name or activity_name.startswith("Hoat dong CTXH ma"):
                activity_name = catalog_item.get("activity_name") or f"Hoạt động CTXH mã {activity_code}"

            raw_pts = row.get("points") or catalog_item.get("points") or 5
            try:
                points = int(float(str(raw_pts).strip().replace(",", ".")))
            except (ValueError, TypeError):
                points = 5

            raw_days = row.get("converted_days") or catalog_item.get("converted_days") or (points / 10)
            try:
                converted_days = float(str(raw_days).strip().replace(",", "."))
            except (ValueError, TypeError):
                converted_days = round(points / 10, 2)

            status = str(row.get("status", "CONFIRMED")).strip().upper() or "CONFIRMED"
            note = str(row.get("note", "")).strip()
            reg_date = str(row.get("registration_date", datetime.now().strftime("%Y-%m-%d"))).strip()
            try:
                float(reg_date)
                reg_date = datetime.now().strftime("%Y-%m-%d")
            except ValueError:
                pass

            reg_id = str(row.get("reg_id", "")).strip()
            if not reg_id:
                short_id = uuid.uuid4().hex[:6].upper()
                reg_id = f"REG-CTXH-{mssv}-{short_id}"

            # Tranh trung lap hoat dong da co
            if reg_id in existing_reg_ids or (activity_code and (activity_code, reg_date) in existing_reg_keys):
                continue

            rec.setdefault("registered_activities", []).append({
                "reg_id": reg_id,
                "activity_code": activity_code,
                "activity_name": activity_name,
                "points": points,
                "converted_days": converted_days,
                "status": status,
                "registration_date": reg_date,
                "note": note,
            })
            existing_reg_ids.add(reg_id)
            if activity_code:
                existing_reg_keys.add((activity_code, reg_date))
            total_new_activities += 1

        # Tinh lai tong so ngay tich luy thuc te theo chuan quy tac lam tron HCMUTE (dong bo voi get_ctxh_status)
        tot_points = sum(int(_parse_numeric_value(a.get("points", 0))) for a in rec.get("registered_activities", []) if a.get("status") == "CONFIRMED")
        rec["accumulated_days"] = convert_points_to_ctxh_days(tot_points)

    _save_student_ctxh(list(records_by_mssv.values()))
    msg = (
        f"Dong bo thanh cong tu '{source_file.name}': "
        f"{len(activities_by_mssv)} sinh vien duoc cap nhat, them {total_new_activities} hoat dong moi vao CSDL."
    )
    sys.stderr.write(f"[sync_ctxh] {msg}\n")
    return {
        "status": "success",
        "file": source_file.name,
        "total_students": len(records_by_mssv),
        "total_activities": sum(len(r.get("registered_activities", [])) for r in records_by_mssv.values()),
    }


def _load_students() -> list[dict]:
    if not STUDENTS_DB_PATH.exists():
        return []
    try:
        with open(STUDENTS_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _load_rules_catalog() -> dict[str, dict]:
    """Nap toan bo danh muc hoat dong CTXH tu activities.json."""
    catalog: dict[str, dict] = {}
    for p in ACTIVITIES_RULES_PATHS:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    for act in cdata.get("activities", []):
                        acode = str(act.get("activity_code", "")).strip()
                        if acode:
                            catalog[acode] = act
            except Exception:
                pass
    return catalog


def _parse_numeric_value(val_str: Any, default: float = 0.0) -> float:
    """Boc tach gia tri so tu chuoi (vi du '0,5', '5->10', '25→50' -> lay so goc)."""
    if val_str is None:
        return default
    if isinstance(val_str, (int, float)):
        return float(val_str)
    s = str(val_str).replace(",", ".").strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return default
    return default


def _normalize_search_string(text: str) -> str:
    """Loai bo dau cau (ngoac don, phay, cham, gach ngang...) de so khop tu vung khong bi lech ky tu."""
    s = re.sub(r"[()\[\]\-:;,.\"/\\?!]", " ", (text or "").lower())
    return " ".join(s.split())


def _parse_points_range(points_str: Any) -> tuple[int, int]:
    """Phan tich khoang diem tu quy che (vi du '5->10', '20→40', '5' -> (min_pts, max_pts))."""
    if points_str is None:
        return 5, 5
    s = str(points_str).replace(",", ".").strip()
    nums = [int(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", s)]
    if len(nums) >= 2:
        return min(nums[0], nums[1]), max(nums[0], nums[1])
    elif len(nums) == 1:
        return nums[0], nums[0]
    return 5, 5




# Initialize and sync store on startup
_ensure_index_loaded()


# ==========================================
# MCP TOOLS (Declared for AI Agents to call)
# ==========================================

@mcp.tool()
def search_ctxh_rules(
    query: str,
    top_k: int = 3,
    year: Optional[str] = None,
) -> str:
    """Tra cuu van ban quy dinh Cong tac xa hoi (CTXH) bang Hybrid Search (BM25 + Dense Vector).
    Tra ve cac dieu khoan, ma hoat dong (6 chu so), thang diem, cach quy doi ngay CTXH, va dieu kien tot nghiep.
    Bat buoc goi cong cu nay khi tra cuu ma hoat dong, quy dinh ngay CTXH hoac tieu chuan mien giam.

    Args:
        query: Tu khoa, ma hoat dong (001101, 001202...), hoac cau hoi can tra cuu quy dinh CTXH.
        top_k: So luong trich doan can lay (mac dinh 3, toi da 6).
        year: Nam ban hanh quy dinh cu the (vi du '2023', '2024', '2025'). Neu de trong, he thong tu dong uu tien ban moi nhat.
    """
    mode = os.getenv("SEARCH_MODE", "hybrid")
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


@mcp.tool()
def get_activity(code_or_name: str) -> str:
    """[READ TOOL] Tra cuu danh muc hoat dong CTXH chinh xac theo ma 6 chu so hoac tu khoa ten hoat dong tu bang quy che.
    Tra ve ma hoat dong, ten day du, linh vuc, thang diem chuan, va so ngay quy doi theo quy dinh kem trich dan nguon chuan.
    Dac biet giup phan biet ro cac hoat dong de gay nham lan (vi du: to chuc hien mau 001201 vs truc tiep hien mau 001202).
    """
    code_or_name = (code_or_name or "").strip()
    if not code_or_name:
        return "Error: Vui long cung cap ma hoat dong hoac tu khoa ten hoat dong can tra cuu."

    catalog = _load_rules_catalog()
    matched = []

    # 1. Exact match by code
    if code_or_name in catalog:
        matched.append(catalog[code_or_name])
    else:
        # Search by keyword / normalized words (loai bo dau ngoac nhu (01 lan) vs 01 lan)
        q_norm = _normalize_search_string(code_or_name)
        STOPWORDS = {
            "hoạt", "động", "tham", "gia", "các", "tại", "cho", "của", "và", "trong", "với", "là", "được", "có",
            "mã", "gì", "bao", "nhiêu", "ngày", "điểm", "nào", "số", "công", "tác", "xã", "hội", "ctxh", "em", "tôi", "muốn"
        }
        q_tokens = [w for w in q_norm.split() if len(w) > 1 and w not in STOPWORDS]

        for c, act in catalog.items():
            name_raw = act.get("activity_name", "")
            cat_raw = act.get("category", "")
            combined_raw = f"{c} {name_raw} {cat_raw}".lower()
            combined_norm = _normalize_search_string(f"{c} {name_raw} {cat_raw}")

            # Khop chuoi con nguyen ban hoac khop cum tu da chuan hoa
            if code_or_name.lower() in combined_raw or (len(q_norm) >= 6 and q_norm in combined_norm):
                matched.append(act)
            elif q_tokens:
                matched_tokens = sum(1 for tok in q_tokens if tok in combined_norm)
                ratio = matched_tokens / len(q_tokens)
                if ratio >= 0.8:
                    matched.append(act)

    if not matched:
        return f"Error: Khong tim thay hoat dong CTXH nao phu hop voi '{code_or_name}' trong danh muc quy che."

    lines = [f"[DANH MUC HOAT DONG CTXH] Tim thay {len(matched)} hoat dong phu hop:"]
    for act in matched[:5]:
        code = act.get("activity_code")
        # Tra cuu chunk that trong kho tri thuc index.json (STORE.chunks) uu tien ban moi nhat
        matching_chunks = [c for c in STORE.chunks.values() if code in c.content]
        if matching_chunks:
            from src.retrieval.vector_store import parse_version_key
            matching_chunks.sort(
                key=lambda c: parse_version_key((c.metadata or {}).get("version")),
                reverse=True
            )
            real_chunk = matching_chunks[0]
            src = real_chunk.metadata.get("file_name", "00_Toan_van_Quy_dinh_CTXH_2025.pdf")
            ver = real_chunk.metadata.get("version", "2025")
            cid = real_chunk.chunk_id
            cit_line = f"[Nguồn: {src} | Phiên bản: {ver} | Mã hoạt động: {code} | Chunk ID: {cid}]"
        else:
            cit_line = f"[Nguồn: Quy chế CTXH HCMUTE (Bản mới nhất) | Mã hoạt động: {code}]"

        lines.append(
            f"- Ma hoat dong: {code}\n"
            f"  + Ten hoat dong: {act.get('activity_name')}\n"
            f"  + Linh vuc: {act.get('category', 'Chung')}\n"
            f"  + Thang diem chuan: {act.get('points')} diem\n"
            f"  + Quy doi ngay CTXH: {act.get('converted_days')} ngay (Cong thuc: {act.get('formula', 'Diem / 10')})\n"
            f"  + Trích dẫn nguồn: {cit_line}"
        )
    return "\n".join(lines)


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
    Bat buoc goi cong cu nay de xac minh danh tinh sinh vien truoc khi xu ly ho so hoac dang ky."""
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
    return f"Error: Khong tim thay sinh vien co MSSV '{mssv}' trong co so du lieu sinh vien cua truong HCMUTE."


@mcp.tool()
def check_graduation_eligibility(mssv: str) -> str:
    """[SYNTHESIS/CALCULATION TOOL] Tham dinh va tinh toan tien do tich luy ngay CTXH so voi chuan tot nghiep.
    Tuong duong get_ctxh_status."""
    return get_ctxh_status(mssv)


@mcp.tool()
def register_ctxh_activity(
    mssv: str,
    activity_code: str,
    points: Optional[int] = None,
    note: str = "",
) -> str:
    """[WRITE TOOL - SIDE EFFECT] Dang ky hoat dong CTXH va ghi nhan vao ho so sinh vien trong he thong du lieu.
    Tu dong kiem tra MSSV ton tai, kiem tra ma hoat dong hop le, kiem tra thang diem hop le theo quy che,
    va kiem tra chong dang ky trung lap.
    Bat buoc goi cong cu get_ctxh_status ngay sau khi goi tool nay de verify trang thai ho so."""
    # 1. Kiem tra MSSV trong he thong goc
    students_master = _load_students()
    master_info = next((s for s in students_master if s.get("mssv") == mssv), None)
    if not master_info:
        return f"Error: Sinh vien co MSSV '{mssv}' khong ton tai trong he thong co so du lieu truong HCMUTE. Khong the dang ky hoat dong."

    # 2. Kiem tra ma hoat dong trong catalog quy che
    catalog = _load_rules_catalog()
    if activity_code not in catalog:
        return f"Error: Ma hoat dong '{activity_code}' khong ton tai trong quy che CTXH hien hanh."

    act_info = catalog[activity_code]
    min_pts, max_pts = _parse_points_range(act_info.get("points"))

    if points is not None:
        if points < min_pts or points > max_pts:
            return (
                f"Error: So diem dang ky {points} khong hop le cho ma hoat dong '{activity_code}'. "
                f"Thang diem quy dinh trong quy che la: {min_pts} den {max_pts} diem."
            )
        points_val = points
    else:
        # Neu khong chi dinh diem, lay mac dinh muc co so trong quy che
        points_val = min_pts

    days_val = convert_points_to_ctxh_days(points_val)

    # 3. Kiem tra ho so sinh vien trong student_activities.json
    data = _load_student_ctxh()
    student = next((s for s in data if s.get("mssv") == mssv), None)
    if not student:
        student = {
            "mssv": mssv,
            "student_name": master_info.get("student_name", "Unknown"),
            "faculty": master_info.get("faculty", ""),
            "course_year": master_info.get("course_year", "2023"),
            "required_days": master_info.get("required_ctxh_days", 4.0),
            "accumulated_days": 0.0,
            "registered_activities": [],
        }
        data.append(student)

    # 4. Kiem tra chong dang ky trung lap hoat dong da xac nhan trong cung ngay
    today_str = datetime.now().strftime("%Y-%m-%d")
    for existing_act in student.get("registered_activities", []):
        if (
            existing_act.get("activity_code") == activity_code
            and existing_act.get("status") == "CONFIRMED"
            and existing_act.get("registration_date") == today_str
        ):
            return (
                f"Error: Sinh vien {mssv} ({student.get('student_name')}) da dang ky hoat dong ma '{activity_code}' "
                f"trong ngay hom nay ({today_str}) voi ma dang ky '{existing_act.get('reg_id')}'. "
                f"He thong khong cho phep dang ky trung lap cung mot hoat dong trong cung mot ngay."
            )

    # 5. Ghi nhan ban ghi moi su dung Dataclass CTXHRegistration
    now_str = datetime.now().strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:4].upper()
    reg_id = f"REG-CTXH-{now_str}-{short_id}"

    reg_obj = CTXHRegistration(
        reg_id=reg_id,
        activity_code=activity_code,
        activity_name=act_info.get("activity_name", f"Hoat dong CTXH ma {activity_code}"),
        points=points_val,
        converted_days=days_val,
        status="CONFIRMED",
        registration_date=today_str,
        note=note or "Dang ky thanh cong qua Tro ly CTXH",
    )
    new_reg = reg_obj.to_dict()
    new_reg["category"] = act_info.get("category", "")
    student.setdefault("registered_activities", []).append(new_reg)

    # Tinh lai tong so ngay tich luy tu tong diem cac hoat dong CONFIRMED theo quy tac lam tron quy che
    confirmed_acts = [a for a in student.get("registered_activities", []) if a.get("status") == "CONFIRMED"]
    total_pts = sum(int(_parse_numeric_value(a.get("points", 0))) for a in confirmed_acts)
    student["accumulated_days"] = convert_points_to_ctxh_days(total_pts)
    _save_student_ctxh(data)

    range_note = ""
    if min_pts != max_pts:
        if points is None:
            range_note = f"\n- Luu y: Hoat dong co khung diem tu {min_pts} den {max_pts} diem. Vi chua co so diem cu the, he thong mac dinh ghi nhan muc co so {min_pts} diem ({days_val} ngay)."
        else:
            range_note = f"\n- Luu y: So diem ghi nhan {points_val} diem nam trong khung quy che {min_pts} den {max_pts} diem."

    return (
        f"[DANG KY HOAT DONG THANH CONG]\n"
        f"- Ma dang ky: {reg_id}\n"
        f"- Sinh vien: {student.get('student_name')} (MSSV: {mssv})\n"
        f"- Hoat dong: {act_info.get('activity_name')} (Ma: {activity_code})\n"
        f"- Linh vuc: {act_info.get('category', 'Chung')}\n"
        f"- Diem ghi nhan: {points_val} diem | So ngay quy doi: +{days_val} ngay\n"
        f"- Tong ngay CTXH tich luy hien tai: {student['accumulated_days']}/{student.get('required_days', 4.0)} ngay (Tong diem: {total_pts} diem)\n"
        f"- Trang thai: CONFIRMED{range_note}\n"
        f"Hay goi ngay get_ctxh_status('{mssv}') de xac thuc va kiem tra tien do ho so."
    )


def convert_points_to_ctxh_days(total_points: float | int) -> float:
    """
    Quy doi tong diem CTXH sang so ngay CTXH theo quy che HCMUTE:
    Lấy tổng điểm chia cho 10:
    - Số lẻ dưới 5 làm tròn xuống (ví dụ 14 điểm -> 1.0 ngày).
    - Tròn 5 điểm quy đổi thành nửa ngày (ví dụ 5 điểm -> 0.5 ngày; 15 điểm -> 1.5 ngày; 35 điểm -> 3.5 ngày).
    - Trên 5 làm tròn thành 10 quy đổi thành một ngày (ví dụ 17 điểm -> 2.0 ngày).
    """
    pts = int(round(total_points))
    tens = pts // 10
    rem = pts % 10
    if rem == 0:
        return float(tens)
    elif rem < 5:
        return float(tens)
    elif rem == 5:
        return float(tens) + 0.5
    else:
        return float(tens + 1)


@mcp.tool()
def get_ctxh_status(mssv: str) -> str:
    """[VERIFY READ TOOL] Kiem tra, doi chieu tien do va xac thuc ho so CTXH thuc te cua sinh vien.
    Doi chieu giua ho so goc (students.json) va lich su hoat dong thuc te (student_activities.json).
    Tra ve day du: Thong tin sinh vien, Tong diem tich luy, So ngay da quy doi, So ngay con thieu,
    Ket luan tham dinh tot nghiep (DA DU / CHUA DU DIEU KIEN), va Danh sach hoat dong da ghi nhan."""
    students_master = _load_students()
    master_info = next((s for s in students_master if s.get("mssv") == mssv), None)

    data = _load_student_ctxh()
    student = next((s for s in data if s.get("mssv") == mssv), None)

    if not student and not master_info:
        return f"Error: Khong tim thay ho so sinh vien co MSSV '{mssv}' trong he thong."

    name = (master_info or {}).get("student_name") or (student or {}).get("student_name", "N/A")
    faculty = (master_info or {}).get("faculty") or (student or {}).get("faculty", "N/A")
    course_year = (master_info or {}).get("course_year") or (student or {}).get("course_year", "N/A")
    req_days = (master_info or {}).get("required_ctxh_days") or (student or {}).get("required_days", 4.0)

    activities = (student or {}).get("registered_activities", [])
    confirmed_acts = [a for a in activities if a.get("status") == "CONFIRMED"]

    total_points = sum(int(_parse_numeric_value(a.get("points", 0))) for a in confirmed_acts)
    acc_days = convert_points_to_ctxh_days(total_points)
    req_days = float(req_days)

    diff = round(req_days - acc_days, 2)
    req_points_est = int(req_days * 10)
    diff_points = max(0, req_points_est - total_points)

    if diff <= 0:
        progress_status = "DA DU DIEU KIEN TOT NGHIEP CTXH"
        detail_msg = f"Sinh vien da hoan tat {acc_days}/{req_days} ngay CTXH (Tong diem tich luy: {total_points} diem)."
    else:
        progress_status = f"CHUA DU DIEU KIEN TOT NGHIEP (CON THIEU {diff} NGAY)"
        detail_msg = (
            f"Sinh vien hien co {total_points} diem CTXH, quy doi duoc {acc_days}/{req_days} ngay CTXH. "
            f"Can tich luy them it nhat {diff} ngay nua (tuong duong khoang {diff_points} diem hoat dong)."
        )

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
        f"[XAC THUC HO SO CTXH & DIEU KIEN TOT NGHIEP]\n"
        f"- MSSV: {mssv} | Ho ten: {name}\n"
        f"- Khoa: {faculty} | Khoa hoc: {course_year}\n"
        f"- Tong diem CTXH tich luy: {total_points} diem\n"
        f"- So ngay tich luy thuc te: {acc_days} ngay (Cong thuc: Tong diem / 10)\n"
        f"- So ngay tieu chuan can tich luy: {req_days} ngay\n"
        f"- Danh gia tien do tot nghiep: {progress_status}\n"
        f"- Chi tiet: {detail_msg}\n"
        f"- Danh sach hoat dong da ghi nhan:\n{acts_text}"
    )



@mcp.tool()
def sync_ctxh_activities() -> str:
    """[ADMIN TOOL] Dong bo du lieu hoat dong da tham gia CTXH cua sinh vien tu file Excel trong data/ vao student_activities.json.
    Bao toan 100% lich su hoat dong hien co cua sinh vien."""
    try:
        result = sync_ctxh_from_file()
    except Exception as e:
        return f"Loi khi dong bo hoat dong CTXH: {e}"

    if result["status"] == "success":
        return (
            f"[CTXH SYNC - THANH CONG] Da dong bo tu file '{result['file']}'.\n"
            f"  - Tong so sinh vien co hoat dong: {result['total_students']}\n"
            f"  - Tong so ban ghi hoat dong: {result['total_activities']}\n"
            f"  - Du lieu da duoc ghi vao student_activities.json."
        )
    elif result["status"] in {"none", "kept"}:
        return f"[CTXH SYNC - BAO TOAN] {result['message']}"
    elif result["status"] == "empty":
        return f"[CTXH SYNC - TRONG] {result['message']}"
    else:
        return f"[CTXH SYNC - LOI] {result.get('message', 'Loi khong xac dinh.')}"


@mcp.tool()
def export_ctxh_rules_json() -> str:
    """[ADMIN TOOL] Boc tach quy dinh, ma hoat dong (6 chu so), thang diem va quy tac quy doi tu cac tai lieu quy che van ban vao activities.json."""
    try:
        result = sync_ctxh_rules_from_file()
    except Exception as e:
        return f"Loi khi boc tach quy che CTXH: {e}"

    if result["status"] == "success":
        return (
            f"[CTXH RULES - THANH CONG] Da boc tach du lieu tu cac file '{result['file']}' vao '{result['target_file']}'.\n"
            f"  - Tong so hoat dong quy doi: {result['total_activities']}\n"
            f"  - Bao gom: Ma hoat dong (6 chu so), linh vuc, thang diem, quy doi ngay va quy tac tinh diem.\n"
            f"  - File ket qua: data/{result['target_file']}"
        )
    elif result["status"] == "empty":
        return f"[CTXH RULES - TRONG] {result['message']}"
    else:
        return f"[CTXH RULES - LOI] {result.get('message', 'Loi khong xac dinh.')}"


@mcp.tool()
def sync_documents() -> str:
    """Dong bo hoa va quet lai thu muc data/ de cap nhat cac file quy dinh CTXH, CSDL sinh vien tu Excel, hoat dong CTXH, va file activities.json."""
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
        lines.append("  - CSDL sinh vien: Khong tim thay tep students.xlsx. Danh sach sinh vien duoc giu nguyen.")
    elif excel_report.get("status") == "empty":
        lines.append("  - CSDL sinh vien: Tep students.xlsx trong, khong co du lieu.")
    elif excel_report.get("status") == "error":
        lines.append(f"  - CSDL sinh vien: {excel_report.get('message')}")

    # Hoat dong CTXH tu file Excel nguon (neu co file thi thong bao, neu khong co thi khoi thong bao)
    if ctxh_report.get("status") == "success":
        lines.append(
            f"  - Hoat dong CTXH (sinh vien): Dong bo thanh cong {ctxh_report.get('total_students')} sinh vien, "
            f"{ctxh_report.get('total_activities')} hoat dong tu '{ctxh_report.get('file')}'."
        )
    elif ctxh_report.get("status") == "error":
        lines.append(f"  - Hoat dong CTXH (sinh vien): {ctxh_report.get('message')}")

    # Quy che va cach tinh diem CTXH da boc tach vao activities.json / ctxh.json
    if rules_report.get("status") == "success":
        lines.append(
            f"  - Bang diem & Quy tac CTXH: Da boc tach thanh cong {rules_report.get('total_activities')} hoat dong va cach tinh diem tu '{rules_report.get('file')}' vao activities.json."
        )
    elif rules_report.get("status") == "empty":
        lines.append(f"  - Bang diem & Quy tac CTXH: {rules_report.get('message')}")
    elif rules_report.get("status") == "error":
        lines.append(f"  - Bang diem & Quy tac CTXH: {rules_report.get('message')}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")

