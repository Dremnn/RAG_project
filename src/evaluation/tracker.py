"""
Module Session Telemetry & Tracking cho RAG Client (Week 5 & Week 7).
Tách riêng toàn bộ trách nhiệm:
- Ghi nhận trace ReAct Agent hàng ngày (run_YYYYMMDD.jsonl)
- Thu thập metrics phiên tương tác (latency, token usage, citations)
- Xuất báo cáo CSV chuẩn utf-8-sig
- Tự động vẽ và xuất biểu đồ DPI 300 vào logs/app/sessions/
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.evaluation.reporter import save_and_plot_session


class SessionTracker:
    """
    Theo dõi telemetry và xuất báo cáo CSV / Đồ thị cho phiên làm việc của RAG Client.
    """

    def __init__(self, logs_dir: Path, session_id: Optional[str] = None):
        self.logs_dir = Path(logs_dir)
        self.sessions_dir = self.logs_dir / "sessions"
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records: list[dict[str, Any]] = []

    def log_agent_run(self, run_record: dict[str, Any]) -> None:
        """Lưu trace từng lượt gọi agent vào logs/app/run_YYYYMMDD.jsonl."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now().strftime("%Y%m%d")
        log_file = self.logs_dir / f"run_{today_str}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record, ensure_ascii=False) + "\n")

    def record_turn(
        self,
        query: str,
        answer: str,
        duration: float,
        usage: dict[str, Any],
        category: str = "interactive",
    ) -> dict[str, Any]:
        """Tạo và lưu trữ record chỉ số của một lượt hỏi đáp."""
        has_cit = (
            ("[nguồn:" in answer.lower() or "[nguon:" in answer.lower())
            and "chunk id:" in answer.lower()
        )
        record = {
            "id": f"Q{len(self.records) + 1}",
            "category": category,
            "query": query,
            "final_answer": answer,
            "answer_preview": answer[:150],
            "duration_seconds": round(duration, 2),
            "has_citation": has_cit,
            "faithfulness_score": 1.0,
            "keyword_recall": 1.0,
            "calls": usage.get("api_calls", 1),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "max_prompt": usage.get("max_prompt", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        self.records.append(record)
        return record

    def update_last_faithfulness(self, score: float) -> None:
        """Cập nhật điểm Faithfulness vào câu hỏi gần nhất sau khi thẩm định /eval."""
        if self.records:
            self.records[-1]["faithfulness_score"] = score

    def export_report(self, is_manual: bool = False) -> Optional[dict[str, Any]]:
        """Xuất file CSV và vẽ đồ thị DPI 300 cho phiên hiện tại."""
        if not self.records:
            return None

        report_out = save_and_plot_session(
            records=self.records,
            output_dir=self.sessions_dir,
            session_id=self.session_id,
            session_title=f"Phiên tương tác {self.session_id}",
        )
        return report_out

    def is_empty(self) -> bool:
        """Kiểm tra xem phiên đã có câu hỏi nào chưa."""
        return len(self.records) == 0

    def get_summary(self) -> dict[str, Any]:
        """Tổng kết số token và lượt hỏi đáp của phiên."""
        tot_tok = sum(r.get("total_tokens", 0) for r in self.records)
        return {
            "session_id": self.session_id,
            "total_questions": len(self.records),
            "total_tokens": tot_tok,
        }
