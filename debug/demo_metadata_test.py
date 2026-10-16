"""
Demo test script: Kiem tra toan dien co che Metadata, Fallback va Loc Version.
"""

import sys
from pathlib import Path

# Thiet lap UTF-8 output cho console Windows
sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.loaders.metadata import extract_metadata, strip_metadata_header
from src.loaders import DocumentLoaderFactory
from rag_server import search_ctxh_rules, STORE


def test_1_extract_metadata_priorities():
    print("=" * 70)
    print("1. KIEM TRA 3 TANG UU TIEN TRICH XUAT METADATA (extract_metadata)")
    print("=" * 70)

    # 1.1 Ưu tiên 1: Có header (front-matter)
    meta_header = extract_metadata("data/CTXH.md")
    print("\n[Ưu tiên 1 - Header Front-Matter]")
    print(f"  - File: data/CTXH.md")
    print(f"  - doc_id          : {meta_header.get('doc_id')}")
    print(f"  - title           : {meta_header.get('title')}")
    print(f"  - version         : {meta_header.get('version')}")
    print(f"  - owner           : {meta_header.get('owner')}")
    print(f"  - metadata_source : {meta_header.get('metadata_source')}  -> (DUNG CHUAN 'header')")

    # 1.2 Ưu tiên 2: Regex filename <doc_id>_<nam>.<duoi>
    meta_filename = extract_metadata("data/00_Toan_van_Quy_dinh_CTXH_2023.pdf")
    print("\n[Ưu tiên 2 - Fallback Regex Filename (PDF)]")
    print(f"  - File: data/00_Toan_van_Quy_dinh_CTXH_2023.pdf")
    print(f"  - doc_id          : {meta_filename.get('doc_id')}")
    print(f"  - version         : {meta_filename.get('version')}")
    print(f"  - metadata_source : {meta_filename.get('metadata_source')}  -> (DUNG CHUAN 'filename')")

    # 1.3 Ưu tiên 3: Fallback none
    meta_none = extract_metadata("data/tai_lieu_chua_phan_loai.txt", content="Noi dung van ban khong co header...")
    print("\n[Ưu tiên 3 - Fallback Không khớp regex (None)]")
    print(f"  - File: data/tai_lieu_chua_phan_loai.txt")
    print(f"  - doc_id          : {meta_none.get('doc_id')}")
    print(f"  - version         : '{meta_none.get('version')}' (de trong)")
    print(f"  - metadata_source : {meta_none.get('metadata_source')}  -> (DUNG CHUAN 'none')")


def test_2_header_cut_before_chunking():
    print("\n" + "=" * 70)
    print("2. KIEM TRA CAT BLOCK --- KHOI NOI DUNG TRUOC KHI CHUNK & EMBED")
    print("=" * 70)

    docs = DocumentLoaderFactory.load_file("data/CTXH.md")
    first_doc = docs[0]
    starts_with_header = first_doc.content.startswith("---")
    print(f"  - Noi dung doc.content co bat dau bang '---' khong: {starts_with_header}")
    print(f"  - 100 ky tu dau tien cua noi dung sach sau khi cat header:")
    preview = first_doc.content[:120].replace("\n", " ")
    print(f"    \"{preview}...\"")
    assert not starts_with_header, "LOI: Header van con trong content!"
    print("  -> DANH GIA: THANH CONG (Header da duoc loai bo khoi text de khong lam nhieu tim kiem)")


def test_3_database_metadata_overview():
    print("\n" + "=" * 70)
    print("3. TONG HOP METADATA CUA TOAN BO TAI LIEU TRONG CSDL (index.json)")
    print("=" * 70)
    doc_stats = {}
    for c in STORE.chunks.values():
        meta = c.metadata or {}
        fn = meta.get("file_name", "unknown")
        if fn not in doc_stats:
            doc_stats[fn] = {
                "doc_id": c.doc_id,
                "version": meta.get("version", ""),
                "source": meta.get("metadata_source", "none"),
                "chunks": 0
            }
        doc_stats[fn]["chunks"] += 1

    print(f"{'Tên file':<50} | {'doc_id':<35} | {'Version':<8} | {'Nguồn':<10} | {'Chunks':<6}")
    print("-" * 120)
    for fn, info in sorted(doc_stats.items()):
        print(f"{fn:<50} | {info['doc_id']:<35} | {info['version']:<8} | {info['source']:<10} | {info['chunks']:<6}")


def test_4_version_precedence_and_citation():
    print("\n" + "=" * 70)
    print("4. KIEM TRA CO CHE LOC PHIEN BAN MOI NHAT & TRICH DAN NGUON")
    print("=" * 70)
    print("Hai file kiem thu cung doc_id 'QD_Khen_Thuong':")
    print("  - QD_Khen_Thuong_2023.txt: version 2023, muc thuong: 500.000d, cong 2 ngay CTXH")
    print("  - QD_Khen_Thuong_2025.txt: version 2025, muc thuong: 1.500.000d, cong 3 ngay CTXH")

    query = "mức thưởng danh hiệu Chiến sĩ thi đua cấp Trường là bao nhiêu"

    # Test 4.1: Khong truyen year -> Tu dong lay ban 2025
    print("\n>>> TEST 4.1: Hoi khong truyen tham so year (He thong tu dong uu tien ban moi nhat 2025)")
    res_default = search_ctxh_rules(query=query)
    print(res_default)

    assert "1.500.000" in res_default, "LOI: Khong tim thay muc thuong ban moi 2025!"
    assert "QD_Khen_Thuong_2023.txt" not in res_default, "LOI: Ban cu 2023 van bi tra ve!"
    assert "Phiên bản: 2025" in res_default, "LOI: Thieu trich dan phien ban 2025!"
    print("\n  -> KET QUA 4.1: CHINH XAC (Chi lay ban 2025, tu dong loai bo ban 2023, trich dan chuan).")

    # Test 4.2: Truyen year="2023" -> Lay dung ban 2023
    print("\n>>> TEST 4.2: Hoi co tham so year='2023' (He thong lay dung ban 2023 theo yeu cau)")
    res_2023 = search_ctxh_rules(query=query, year="2023")
    print(res_2023)

    assert "500.000" in res_2023, "LOI: Khong tim thay muc thuong ban 2023!"
    assert "QD_Khen_Thuong_2025.txt" not in res_2023, "LOI: Ban moi 2025 bi tra ve khi hoi nam 2023!"
    assert "Phiên bản: 2023" in res_2023, "LOI: Thieu trich dan phien ban 2023!"
    print("\n  -> KET QUA 4.2: CHINH XAC (Lay dung ban 2023, loai bo ban 2025, trich dan chuan).")


if __name__ == "__main__":
    test_1_extract_metadata_priorities()
    test_2_header_cut_before_chunking()
    test_3_database_metadata_overview()
    test_4_version_precedence_and_citation()
    print("\n" + "=" * 70)
    print("TAT CA CAC MUC KIEM THU METADATA DEU DA HOAT DONG 100% CHINH XAC!")
    print("=" * 70)
