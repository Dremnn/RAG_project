"""
Metadata extraction for document loaders.

Thứ tự ưu tiên trích xuất metadata:
1. Block front-matter (--- ... ---) ở đầu file (.md, .txt, .docx).
2. Regex theo tên file: <doc_id>_<năm>.<đuôi> (áp dụng cho PDF và file không có header).
3. Fallback: doc_id = stem, version = "", metadata_source = "none".
"""

from pathlib import Path
import re
from typing import Any, Optional

FRONT_MATTER_REGEX = re.compile(
    r"^\s*---\r?\n(.*?)\r?\n---\r?\n?",
    re.DOTALL
)

FILENAME_REGEX = re.compile(
    r"^(?P<doc_id>.+)_(?P<version>\d{4}(?:-\d{2}-\d{2})?)$"
)


def strip_metadata_header(content: str) -> str:
    """
    Cắt bỏ block metadata header (front-matter --- ... ---) khỏi nội dung văn bản.
    """
    if not content:
        return ""
    match = FRONT_MATTER_REGEX.match(content)
    if match:
        return content[match.end():].strip()
    return content.strip()


def _parse_yaml_front_matter(raw_yaml: str) -> dict[str, Any]:
    """
    Parse nội dung YAML front-matter an toàn (ưu tiên PyYAML, fallback key-value).
    """
    try:
        import yaml
        parsed = yaml.safe_load(raw_yaml)
        if isinstance(parsed, dict):
            return {str(k).strip(): v for k, v in parsed.items()}
    except Exception:
        pass

    parsed: dict[str, Any] = {}
    for line in raw_yaml.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            parsed[key] = val
    return parsed


def extract_metadata(
    path: str | Path,
    content: Optional[str] = None,
) -> dict[str, Any]:
    """
    Trích xuất metadata của tài liệu theo thứ tự ưu tiên:
    1. Header front-matter (--- ... ---): metadata_source = 'header'
    2. Regex tên file <doc_id>_<năm>.<đuôi>: metadata_source = 'filename'
    3. Fallback: doc_id lấy tên file không đuôi, version để trống: metadata_source = 'none'

    Trả về dictionary chứa các trường:
    - doc_id: str
    - title: str
    - version: str (hoặc "" nếu không có)
    - metadata_source: "header" | "filename" | "none"
    - file_name: str
    - extension: str
    - cleaned_content: str (nội dung đã cắt header nếu có)
    - các trường bổ sung từ front-matter (như owner, ...)
    """
    path_obj = Path(path)
    file_name = path_obj.name
    ext = path_obj.suffix.lower()
    stem = path_obj.stem

    docx_core_props: dict[str, str] = {}

    # Đọc nội dung nếu chưa được truyền vào
    if content is None and path_obj.is_file():
        if ext in [".txt", ".md"]:
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(path_obj, "r", encoding="latin-1") as f:
                        content = f.read()
                except Exception:
                    content = ""
            except Exception:
                content = ""
        elif ext == ".docx":
            try:
                import docx
                doc_obj = docx.Document(str(path_obj))
                parts = [p.text.strip() for p in doc_obj.paragraphs if p.text.strip()]
                content = "\n\n".join(parts)
                # Đọc document properties nếu có
                if hasattr(doc_obj, "core_properties"):
                    cp = doc_obj.core_properties
                    if cp.title:
                        docx_core_props["title"] = cp.title.strip()
                    if cp.author:
                        docx_core_props["owner"] = cp.author.strip()
            except Exception:
                content = ""

    # 1. ƯU TIÊN 1: Có block --- ở đầu file
    if content:
        match = FRONT_MATTER_REGEX.match(content)
        if match:
            raw_fm = match.group(1)
            parsed = _parse_yaml_front_matter(raw_fm)

            doc_id = str(parsed.get("doc_id", "")).strip()
            version = str(parsed.get("version", "")).strip() if "version" in parsed else ""
            title = str(parsed.get("title", "")).strip()

            # Nếu thiếu doc_id trong header, fallback regex tên file
            if not doc_id:
                fn_match = FILENAME_REGEX.match(stem)
                doc_id = fn_match.group("doc_id") if fn_match else stem

            # Nếu thiếu version trong header, fallback regex tên file
            if not version:
                fn_match = FILENAME_REGEX.match(stem)
                if fn_match:
                    version = fn_match.group("version")

            if not title:
                title = docx_core_props.get("title") or stem.replace("_", " ")

            cleaned_content = content[match.end():].strip()

            meta: dict[str, Any] = {
                "doc_id": doc_id,
                "title": title,
                "version": version,
                "metadata_source": "header",
                "file_name": file_name,
                "extension": ext,
                "cleaned_content": cleaned_content,
            }

            for k, v in parsed.items():
                if k not in meta:
                    meta[k] = v

            return meta

    # 2. ƯU TIÊN 2: Regex theo tên file: <doc_id>_<năm>.<đuôi>
    fn_match = FILENAME_REGEX.match(stem)
    if fn_match:
        doc_id = fn_match.group("doc_id")
        version = fn_match.group("version")
        title = docx_core_props.get("title") or doc_id.replace("_", " ")

        meta = {
            "doc_id": doc_id,
            "title": title,
            "version": version,
            "metadata_source": "filename",
            "file_name": file_name,
            "extension": ext,
            "cleaned_content": content.strip() if content else "",
        }
        if "owner" in docx_core_props:
            meta["owner"] = docx_core_props["owner"]
        return meta

    # 3. ƯU TIÊN 3: Không khớp regex -> version để trống, doc_id lấy tên file không đuôi
    title = docx_core_props.get("title") or stem.replace("_", " ")
    meta = {
        "doc_id": stem,
        "title": title,
        "version": "",
        "metadata_source": "none",
        "file_name": file_name,
        "extension": ext,
        "cleaned_content": content.strip() if content else "",
    }
    if "owner" in docx_core_props:
        meta["owner"] = docx_core_props["owner"]
    return meta
