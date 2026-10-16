import unittest
import tempfile
import csv
from pathlib import Path
from src.evaluation.reporter import save_session_csv, generate_session_charts, save_and_plot_session


class TestSessionReporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)
        self.sample_records = [
            {
                "id": "Q1",
                "category": "calculation",
                "query": "Hoc 4 nam can bao nhieu ngay CTXH?",
                "final_answer": "Sinh vien can 4.0 ngay CTXH [Nguon: 00_Toan_van.docx | Chunk ID: c1].",
                "duration_seconds": 1.25,
                "has_citation": True,
                "faithfulness_score": 1.0,
                "keyword_recall": 1.0,
                "calls": 2,
                "prompt_tokens": 500,
                "completion_tokens": 120,
                "max_prompt": 350,
                "total_tokens": 620,
            },
            {
                "id": "Q2",
                "category": "lookup",
                "query": "Hien mau duoc bao nhieu ngay CTXH?",
                "final_answer": "Hien mau tinh nguyen duoc quy doi 0.5 ngay.",
                "duration_seconds": 0.95,
                "has_citation": False,
                "faithfulness_score": 0.9,
                "keyword_recall": 0.8,
                "calls": 1,
                "prompt_tokens": 300,
                "completion_tokens": 80,
                "max_prompt": 300,
                "total_tokens": 380,
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_session_csv(self):
        csv_file = self.out_dir / "test_session.csv"
        saved_path = save_session_csv(self.sample_records, csv_file)
        self.assertTrue(saved_path.exists())

        # Verify content
        with open(saved_path, "r", encoding="utf-8-sig") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 3)  # Header + 2 rows
            header = reader[0]
            self.assertIn("id", header)
            self.assertIn("total_tokens", header)
            self.assertEqual(reader[1][0], "Q1")
            self.assertEqual(reader[2][0], "Q2")

    def test_generate_session_charts(self):
        charts = generate_session_charts(
            records=self.sample_records,
            output_dir=self.out_dir / "charts",
            prefix="test_",
            session_title="TEST SESSION",
        )
        self.assertGreaterEqual(len(charts), 2)
        for c in charts:
            self.assertTrue(c.exists())
            self.assertGreater(c.stat().st_size, 1000)

    def test_save_and_plot_session(self):
        res = save_and_plot_session(
            records=self.sample_records,
            output_dir=self.out_dir,
            session_id="20261016_test",
            session_title="UNIT TEST RUN",
        )
        self.assertEqual(res["session_id"], "20261016_test")
        self.assertTrue(res["csv_path"].exists())
        self.assertGreaterEqual(len(res["charts"]), 2)


if __name__ == "__main__":
    unittest.main()
