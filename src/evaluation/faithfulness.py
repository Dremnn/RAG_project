from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import re
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
        badge = "🟢 ĐỘ TIN CẬY CAO" if self.is_trustworthy else "🔴 CẢNH BÁO ẢO GIÁC"
        
        lines = [
            f"┌─── 🛡️ BÁO CÁO THẨM ĐỊNH ĐỘ TRUNG THỰC (FAITHFULNESS) ───┐",
            f"│ Điểm số: {percent}% ({self.supported_claims}/{self.total_claims} mệnh đề)  |  Đánh giá: {badge}",
            f"├────────────────────────────────────────────────────────┤",
        ]
        for idx, v in enumerate(self.verdicts, 1):
            icon = "✅" if v.supported else "❌"
            lines.append(f"│ {icon} Mệnh đề {idx}: \"{v.claim}\"")
            if not v.supported:
                lines.append(f"│    ↳ Nhận định: {v.reason}")
        lines.append(f"└────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


class FaithfulnessEvaluator:
    """
    Evaluator for Groundedness & Faithfulness (Week 5 Architecture).
    Uses atomic claim decomposition + LLM-as-a-judge.
    """

    def __init__(self, groq_client: Groq, model: str = "qwen/qwen3.8-27b"):
        self.client = groq_client
        self.model = model

    def decompose_claims(self, answer: str) -> list[str]:
        """Split a complex answer into atomic standalone factual claims."""
        prompt = f"""Hãy phân tách câu trả lời dưới đây thành danh sách các mệnh đề thực tế độc lập (atomic factual claims).
Mỗi mệnh đề phải là một câu khẳng định đơn nhất, rõ ràng, không chứa đại từ mơ hồ.

Câu trả lời:
\"\"\"{answer}\"\"\"

Yêu cầu:
Trả về DUY NHẤT một JSON array chứa các chuỗi, ví dụ:
["Mệnh đề 1", "Mệnh đề 2", "Mệnh đề 3"]
Không viết thêm bất kỳ lời dẫn nào khác ngoài JSON array."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON array using regex if surrounded by markdown blocks
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                claims = json.loads(match.group(0))
                return [str(c).strip() for c in claims if str(c).strip()]
        except Exception:
            pass

        # Fallback: split by lines/sentences
        lines = [s.strip("- *0123456789. ") for s in answer.split("\n") if len(s.strip()) > 15]
        return lines[:6] if lines else [answer.strip()]

    def judge_claim(self, claim: str, context: str) -> ClaimVerdict:
        """Ask LLM judge whether the claim is logically supported by the context."""
        prompt = f"""Bạn là một thẩm phán đánh giá độ trung thực (Groundedness Judge).
Hãy đối chiếu mệnh đề sau với ngữ cảnh được cung cấp từ tài liệu:

Mệnh đề cần kiểm tra:
\"{claim}\"

Ngữ cảnh tài liệu gốc:
\"\"\"{context[:4000]}\"\"\"

Quy tắc phán quyết:
- Đạt (supported = true): Nếu mệnh đề hoàn toàn đúng hoặc được suy ra trực tiếp từ ngữ cảnh.
- Không đạt (supported = false): Nếu mệnh đề không được nhắc tới, bịa đặt, hoặc trái ngược với ngữ cảnh.

Trả về DUY NHẤT một đối tượng JSON:
{{"supported": true/false, "reason": "giải thích ngắn gọn trong 1 câu"}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return ClaimVerdict(
                    claim=claim,
                    supported=bool(data.get("supported", False)),
                    reason=str(data.get("reason", "")),
                )
        except Exception as e:
            return ClaimVerdict(claim=claim, supported=True, reason=f"Lỗi phân tích: {e}")

        return ClaimVerdict(claim=claim, supported=True, reason="Mặc định hỗ trợ.")

    def evaluate(self, answer: str, context: str) -> FaithfulnessReport:
        """
        Evaluate faithfulness score:
        Faithfulness = Supported Claims / Total Claims
        """
        if not answer.strip() or not context.strip():
            return FaithfulnessReport(
                score=1.0, total_claims=0, supported_claims=0, unsupported_claims=0
            )

        claims = self.decompose_claims(answer)
        if not claims:
            return FaithfulnessReport(
                score=1.0, total_claims=0, supported_claims=0, unsupported_claims=0
            )

        verdicts: list[ClaimVerdict] = []
        supported_count = 0

        for claim in claims:
            v = self.judge_claim(claim, context)
            verdicts.append(v)
            if v.supported:
                supported_count += 1

        total = len(claims)
        score = supported_count / total if total > 0 else 1.0

        return FaithfulnessReport(
            score=round(score, 2),
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=total - supported_count,
            verdicts=verdicts,
        )
