"""
Unit and Integration Tests for Document Metadata, Version Precedence, and Search Filtering.
"""

import sys
import unittest
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.loaders.metadata import extract_metadata, strip_metadata_header
from src.loaders import DocumentLoaderFactory
from src.retrieval.vector_store import VectorStore
from rag_server import search_ctxh_rules, STORE


class TestDocumentMetadata(unittest.TestCase):
    """Test priority order of extract_metadata."""

    def test_priority_1_header(self):
        """Ưu tiên 1: Có block --- ở đầu file."""
        sample_text = (
            "---\n"
            "doc_id: RULE_A\n"
            "title: Quy chế Công tác xã hội\n"
            "version: 2023\n"
            "owner: Phòng Công tác sinh viên\n"
            "---\n\n"
            "Nội dung bắt đầu từ đây."
        )
        meta = extract_metadata("data/RULE_A_2020.txt", content=sample_text)
        self.assertEqual(meta["doc_id"], "RULE_A")
        self.assertEqual(meta["title"], "Quy chế Công tác xã hội")
        self.assertEqual(meta["version"], "2023")
        self.assertEqual(meta["owner"], "Phòng Công tác sinh viên")
        self.assertEqual(meta["metadata_source"], "header")

        cleaned = strip_metadata_header(sample_text)
        self.assertEqual(cleaned, "Nội dung bắt đầu từ đây.")
        self.assertNotIn("---", cleaned)

    def test_priority_2_filename_regex(self):
        """Ưu tiên 2: Không có header, regex theo tên file <doc_id>_<năm>.<đuôi>."""
        meta = extract_metadata("data/A_2023.pdf")
        self.assertEqual(meta["doc_id"], "A")
        self.assertEqual(meta["version"], "2023")
        self.assertEqual(meta["metadata_source"], "filename")

    def test_priority_3_fallback_none(self):
        """Ưu tiên 3: Không khớp regex -> version để trống, doc_id lấy tên file không đuôi."""
        meta = extract_metadata("data/huong_dan_su_dung.txt", content="Huong dan he thong...")
        self.assertEqual(meta["doc_id"], "huong_dan_su_dung")
        self.assertEqual(meta["version"], "")
        self.assertEqual(meta["metadata_source"], "none")

    def test_loader_factory_cuts_header(self):
        """Kiểm tra loader factory tự động gọi extract_metadata và cắt header khỏi content."""
        docs = DocumentLoaderFactory.load_file("data/QD_Khen_Thuong_2023.txt")
        self.assertGreater(len(docs), 0)
        doc = docs[0]
        self.assertEqual(doc.metadata.get("metadata_source"), "header")
        self.assertEqual(doc.metadata.get("version"), "2023")
        self.assertEqual(doc.doc_id, "QD_Khen_Thuong")
        self.assertFalse(doc.content.startswith("---"))


class TestVersionFilteringAndPrecedence(unittest.TestCase):
    """
    Test theo yêu cầu:
    Thêm hai file cùng doc_id, khác version và nội dung hơi khác.
    Hỏi đúng chỗ khác nhau thì kết quả phải theo bản mới, trích dẫn đúng phiên bản.
    """

    def test_prefers_latest_version_when_no_year_specified(self):
        """Hỏi chỗ khác biệt mà không chỉ định năm -> tự động lấy bản 2025."""
        res = search_ctxh_rules("mức thưởng danh hiệu Chiến sĩ thi đua cấp Trường")
        # Bản 2025 là 1.500.000 đồng và 3 ngày CTXH
        self.assertIn("1.500.000", res)
        self.assertIn("3 ngày CTXH", res)
        # Không được lấy bản 2023
        self.assertIn("QD_Khen_Thuong_2025.txt", res)
        self.assertNotIn("QD_Khen_Thuong_2023.txt", res)
        self.assertNotIn("mức thưởng là 500.000 đồng", res)
        # Trích dẫn đúng phiên bản 2025
        self.assertIn("Phiên bản: 2025", res)

    def test_explicit_year_retrieves_older_version(self):
        """Khi truyền tham số year='2023' -> lấy đúng bản 2023."""
        res = search_ctxh_rules("mức thưởng danh hiệu Chiến sĩ thi đua cấp Trường", year="2023")
        # Bản 2023 là 500.000 đồng và 2 ngày CTXH
        self.assertIn("500.000", res)
        self.assertIn("2 ngày CTXH", res)
        self.assertNotIn("1.500.000", res)
        self.assertIn("QD_Khen_Thuong_2023.txt", res)
        self.assertNotIn("QD_Khen_Thuong_2025.txt", res)
        self.assertIn("Phiên bản: 2023", res)

    def test_citation_format_matches_requirement(self):
        """Định dạng chuẩn [Nguồn: <file> | Phiên bản: <ver> | Chunk ID: <id>]."""
        res = search_ctxh_rules("hiến máu tình nguyện")
        self.assertIn("[Nguồn:", res)
        self.assertIn("Phiên bản:", res)
        self.assertIn("Chunk ID:", res)


if __name__ == "__main__":
    unittest.main()
