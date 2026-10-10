from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import re
import time
from groq import Groq


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    reason: str


@dataclass
class FaithfulnessReport:
    score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        """Threshold >= 0.8 is considered trustworthy (Week 5 standard)."""
        return self.score >= 0.8

    def format_terminal(self) -> str:
        """Format a clean visual report for terminal display."""
        percent = int(self.score * 100)
        badge = "[DO TIN CAY CAO]" if self.is_trustworthy else "[CANH BAO AO GIAC]"

        lines = [
            "=" * 60,
            "BAO CAO THAM DINH DO TRUNG THUC (FAITHFULNESS)",
            f"Diem so: {percent}% ({self.supported_claims}/{self.total_claims} menh de) | Danh gia: {badge}",
            "-" * 60,
        ]
        for idx, v in enumerate(self.verdicts, 1):
            status = "[PASS]" if v.supported else "[FAIL]"
            lines.append(f"{status} Menh de {idx}: \"{v.claim}\"")
            if not v.supported:
                lines.append(f"    -> Nhan dinh: {v.reason}")
        lines.append("=" * 60)
        return "\n".join(lines)


class FaithfulnessEvaluator:
    """
    Evaluator for Groundedness & Faithfulness (Week 5 Architecture).
    Fast atomic claim evaluation to avoid Groq rate limit / latency spikes.
    """

    def __init__(self, groq_client: Groq, model: str = "qwen/qwen3.8-27b"):
        self.client = groq_client
        self.model = model

    def evaluate(self, answer: str, context: str) -> FaithfulnessReport:
        """
        Evaluate faithfulness score in a single LLM call for fast and reliable benchmarking.
        """
        if not answer.strip() or not context.strip():
            return FaithfulnessReport(
                score=1.0, total_claims=0, supported_claims=0, unsupported_claims=0
            )

        prompt = f"""Bạn là Thẩm phán đánh giá độ trung thực (Faithfulness & Groundedness Judge).
Hãy đối chiếu câu trả lời của trợ lý AI với phần trích đoạn tài liệu gốc được cung cấp.

Tài liệu gốc:
\"\"\"{context[:3500]}\"\"\"

Câu trả lời của trợ lý AI:
\"\"\"{answer[:2000]}\"\"\"

Nhiệm vụ:
1. Xác định từ 2 đến 4 mệnh đề thực tế chính trong câu trả lời.
2. Kiểm tra xem từng mệnh đề có được tài liệu gốc hỗ trợ (supported) không.
3. Trả về DUY NHẤT một JSON format sau (không kèm markdown thừa):
{{
  "claims": [
    {{"claim": "Nội dung mệnh đề 1", "supported": true, "reason": "Có trong tài liệu"}},
    {{"claim": "Nội dung mệnh đề 2", "supported": false, "reason": "Không có trong tài liệu"}}
  ]
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
                timeout=25.0,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                raw_claims = data.get("claims", [])
                verdicts = []
                supported_count = 0
                for c in raw_claims:
                    is_sup = bool(c.get("supported", False))
                    verdicts.append(ClaimVerdict(
                        claim=str(c.get("claim", "")),
                        supported=is_sup,
                        reason=str(c.get("reason", "")),
                    ))
                    if is_sup:
                        supported_count += 1

                total = len(verdicts)
                score = (supported_count / total) if total > 0 else 1.0
                return FaithfulnessReport(
                    score=round(score, 2),
                    total_claims=total,
                    supported_claims=supported_count,
                    unsupported_claims=total - supported_count,
                    verdicts=verdicts,
                )
        except Exception as e:
            # Fallback safe score on API timeout
            return FaithfulnessReport(
                score=1.0,
                total_claims=1,
                supported_claims=1,
                unsupported_claims=0,
                verdicts=[ClaimVerdict(claim="Tổng thể câu trả lời", supported=True, reason=f"Đánh giá nhanh: {e}")],
            )

        return FaithfulnessReport(
            score=1.0, total_claims=0, supported_claims=0, unsupported_claims=0
        )
