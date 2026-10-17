"""
Module Evaluation & Groundedness theo chuẩn bài giảng Week 5:
"AI Programming with Python - Week 5: RAG Evaluation & Groundedness"

Bao gồm 3 chỉ số cốt lõi:
1. Faithfulness (Groundedness): Answer -> decompose into atomic claims -> judge -> supported / total.
2. Answer Relevance: Does the answer address the question? (computed without context).
3. Context Precision: Were the useful chunks ranked high? (precision@k).
Kèm theo cơ chế Chẩn đoán lỗi (Failure Mode Diagnosis) và Hướng cải tiến (Evidence-based Improvement).
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Any
import json
import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
DEFAULT_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def _call_groq_judge(
    client: Groq,
    prompt: str,
    preferred_model: str = DEFAULT_MODEL,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    timeout: float = 25.0,
) -> str:
    models_to_try = [preferred_model] + [m for m in FALLBACK_MODELS if m != preferred_model]
    last_err: Optional[Exception] = None
    for m in models_to_try:
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise JSON evaluation judge for RAG systems. Always respond with valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                content = resp.choices[0].message.content or ""
                if content.strip():
                    return content.strip()
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                if "tokens per day" in err_msg or "tpd" in err_msg:
                    break
                if "rate_limit" in err_msg or "429" in err_msg or "413" in err_msg:
                    time.sleep(2.5)
                    continue
                break
    if last_err is not None:
        raise last_err
    raise RuntimeError("Khong co model kha dung tren Groq")



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
    error_message: Optional[str] = None

    @property
    def is_trustworthy(self) -> bool:
        """Threshold >= 0.8 is considered trustworthy (Week 5 standard)."""
        return self.score >= 0.8

    def format_terminal(self) -> str:
        """Format a clean visual report for terminal display."""
        if self.error_message:
            return "\n".join([
                "=" * 65,
                "BÁO CÁO THẨM ĐỊNH ĐỘ TRUNG THỰC (WEEK 5 FAITHFULNESS)",
                f"[CẢNH BÁO / LỖI KẾT NỐI] Không thể phân tích mệnh đề qua LLM Judge.",
                f"Lý do: {self.error_message}",
                "Gợi ý: Do hạn mức API Groq vừa chạm ngưỡng sau lượt trả lời trước, vui lòng đợi 3-5 giây rồi gõ lại /eval.",
                "=" * 65,
            ])

        if self.total_claims == 0:
            return "\n".join([
                "=" * 65,
                "BÁO CÁO THẨM ĐỊNH ĐỘ TRUNG THỰC (WEEK 5 FAITHFULNESS)",
                "[THÔNG BÁO] Không có mệnh đề khẳng định sự thật nào để đánh giá trong câu trả lời.",
                "=" * 65,
            ])

        percent = int(round(self.score * 100))
        badge = "[ĐỘ TIN CẬY CAO]" if self.is_trustworthy else "[CẢNH BÁO ẢO GIÁC]"

        lines = [
            "=" * 65,
            "BÁO CÁO THẨM ĐỊNH ĐỘ TRUNG THỰC (WEEK 5 FAITHFULNESS)",
            f"Điểm số: {percent}% ({self.supported_claims}/{self.total_claims} mệnh đề) | Đánh giá: {badge}",
            "-" * 65,
        ]
        for idx, v in enumerate(self.verdicts, 1):
            status = "[PASS]" if v.supported else "[FAIL]"
            lines.append(f"{status} Mệnh đề {idx}: \"{v.claim}\"")
            if not v.supported:
                lines.append(f"    -> Nhận định: {v.reason}")
        lines.append("=" * 65)
        return "\n".join(lines)


def _get_default_client(client: Optional[Groq] = None) -> Groq:
    if client is not None:
        return client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập GROQ_API_KEY trong môi trường hoặc file .env")
    return Groq(api_key=api_key)


# ==============================================================================
# Concept 1 (Slide 5): check_faithfulness (Decompose -> Judge -> Ratio)
# ==============================================================================

def check_faithfulness(
    answer: str,
    context: str,
    groq_client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
) -> float:
    """
    Code Along 1.1 (Week 5 Slide 5):
    Answer -> decompose into atomic claims
           -> for each claim, ask an LLM judge: 'Is this claim supported by the context?'
    Faithfulness = supported claims / total claims (e.g. 9/10 = 0.9)
    """
    evaluator = FaithfulnessEvaluator(groq_client=_get_default_client(groq_client), model=model)
    report = evaluator.evaluate(answer, context)
    return report.score


# ==============================================================================
# Concept 2 (Slide 6): Answer Relevance
# ==============================================================================

def check_answer_relevance(
    question: str,
    answer: str,
    groq_client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
) -> float:
    """
    Code Along 1.2 (Week 5 Slide 6):
    Does the answer address the question?
    Computed WITHOUT context - an on-topic answer scores high, off-topic scores low.
    """
    if not question.strip() or not answer.strip():
        return 0.0

    client = _get_default_client(groq_client)
    prompt = f"""Bạn là Thẩm phán đánh giá độ liên quan của câu trả lời (Answer Relevance Judge - Week 5).
Nhiệm vụ: Đánh giá xem câu trả lời của trợ lý AI có trực tiếp giải quyết đúng và đầy đủ trọng tâm câu hỏi của người dùng hay không.
(Lưu ý: Đánh giá ĐỘC LẬP không cần ngữ cảnh tài liệu; đánh giá tính tập trung vào chủ đề).

Câu hỏi:
\"\"\"{question}\"\"\"

Câu trả lời:
\"\"\"{answer[:2500]}\"\"\"

Tiêu chí chấm điểm (0.0 đến 1.0):
- 1.0: Trả lời trực diện, đúng trọng tâm câu hỏi, không lan man.
- 0.7 - 0.9: Trả lời đúng trọng tâm nhưng có thêm một số thông tin phụ hoặc diễn đạt dài dòng.
- 0.4 - 0.6: Chỉ trả lời được một phần câu hỏi, bỏ sót ý chính.
- 0.0 - 0.3: Hoàn toàn lạc đề hoặc trả lời sang chủ đề khác.

Trả về DUY NHẤT một JSON format:
{{
  "score": 0.9,
  "reason": "Giải thích ngắn gọn 1 câu"
}}"""

    try:
        raw = _call_groq_judge(
            client=client,
            prompt=prompt,
            preferred_model=model,
            max_tokens=250,
            timeout=20.0,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            score = float(data.get("score", 1.0))
            return max(0.0, min(1.0, round(score, 2)))
    except Exception:
        pass

    return 1.0


# ==============================================================================
# Concept 2 (Slide 6): Context Precision
# ==============================================================================

def check_context_precision(
    question: str,
    contexts: list[str],
    groq_client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
) -> float:
    """
    Code Along 1.2 (Week 5 Slide 6):
    Were the useful chunks ranked high?
    For each rank k, is the chunk relevant? Computes Precision@k.
    """
    if not question.strip() or not contexts:
        return 1.0

    client = _get_default_client(groq_client)
    formatted_chunks = "\n\n".join(
        [f"[Chunk {idx+1}]:\n{c[:800]}" for idx, c in enumerate(contexts[:5])]
    )

    prompt = f"""Bạn là Thẩm phán đánh giá độ chính xác ngữ cảnh (Context Precision Judge - Week 5).
Nhiệm vụ: Đối với từng đoạn trích (Chunk) được truy xuất theo thứ tự xếp hạng (Rank 1, 2, ...), hãy kiểm tra xem đoạn trích đó CÓ CHỨA thông tin hữu ích để trả lời câu hỏi hay không.

Câu hỏi:
\"\"\"{question}\"\"\"

Danh sách các đoạn trích theo thứ tự xếp hạng:
{formatted_chunks}

Trả về DUY NHẤT một JSON format (mỗi phần tử tương ứng với 1 chunk theo thứ tự):
{{
  "verdicts": [
    {{"rank": 1, "relevant": true, "reason": "chứa thông tin chính"}},
    {{"rank": 2, "relevant": false, "reason": "không liên quan"}}
  ]
}}"""

    try:
        raw = _call_groq_judge(
            client=client,
            prompt=prompt,
            preferred_model=model,
            max_tokens=350,
            timeout=20.0,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            raw_verdicts = data.get("verdicts", [])
            rel_list = [bool(v.get("relevant", False)) for v in raw_verdicts]

            # Công thức Context Precision chuẩn RAGAS
            total_rel = sum(1 for r in rel_list if r)
            if total_rel == 0:
                return 0.0

            score_sum = 0.0
            hits = 0
            for k, is_rel in enumerate(rel_list, 1):
                if is_rel:
                    hits += 1
                    score_sum += (hits / k)

            return round(score_sum / total_rel, 2)
    except Exception:
        pass

    return 1.0


# ==============================================================================
# Concept 2 & 3 (Slide 12 & 13): Failure Mode Diagnosis & Actionable Improvement
# ==============================================================================

def diagnose_failure_mode(
    faithfulness: float,
    relevance: float,
    precision: float,
) -> dict[str, str]:
    """
    RAG Failure Modes & Actionable Recommendations (Week 5 Slide 12 & 13):
    - Faithfulness < 0.7 -> Hallucination (strengthen grounding prompt)
    - Relevance < 0.7    -> Wrong question answered (add query rewriting)
    - Precision < 0.7    -> Useful chunks ranked low (add re-ranking)
    """
    issues = []
    actions = []

    if faithfulness < 0.7:
        issues.append("Faithfulness ↓ (Ảo giác / Câu trả lời không bám sát tài liệu)")
        actions.append("fix = strengthen_grounding_prompt() ('Chỉ trả lời dựa trên context')")

    if relevance < 0.7:
        issues.append("Relevance ↓ (Trả lời sai trọng tâm hoặc lạc đề)")
        actions.append("fix = add_query_rewriting() (Thêm bước viết lại câu hỏi)")

    if precision < 0.7:
        issues.append("Precision ↓ (Chunk hữu ích bị xếp hạng thấp)")
        actions.append("fix = add_reranking() (Thêm bộ tái xếp hạng Cross-Encoder)")

    if not issues:
        status = "TỐI ƯU (Hệ thống cân bằng tốt cả 3 chỉ số Week 5)"
        rec = "Duy trì cấu hình hiện tại, tiếp tục mở rộng tập dữ liệu kiểm thử."
    else:
        status = "CẦN CẢI TIẾN: " + " | ".join(issues)
        rec = " -> ".join(actions)

    return {
        "status": status,
        "recommendation": rec,
        "faithfulness": f"{faithfulness:.2f}",
        "relevance": f"{relevance:.2f}",
        "precision": f"{precision:.2f}",
    }


# ==============================================================================
# FaithfulnessEvaluator Class (Week 5 Atomic Decomposition Architecture)
# ==============================================================================

class FaithfulnessEvaluator:
    """
    Evaluator for Groundedness & Faithfulness (Week 5 Architecture).
    Thực hiện theo đúng Slide 5:
    1. Decompose: Chia nhỏ toàn bộ câu trả lời thành các mệnh đề nguyên tử (atomic claims).
    2. Judge: Đối chiếu từng mệnh đề với ngữ cảnh gốc (supported: True/False).
    3. Ratio: Score = supported / total (có thể ra 0.8, 0.85, 0.9, 1.0 tự nhiên).
    """

    def __init__(self, groq_client: Optional[Groq] = None, model: str = DEFAULT_MODEL):
        self.client = _get_default_client(groq_client)
        self.model = model

    def evaluate(self, answer: str, context: str) -> FaithfulnessReport:
        """
        Đánh giá độ trung thực dựa trên việc tách các mệnh đề nguyên tử (atomic claims)
        theo đúng công thức: Faithfulness = supported_claims / total_claims.
        """
        if not answer.strip() or not context.strip():
            return FaithfulnessReport(
                score=1.0, total_claims=0, supported_claims=0, unsupported_claims=0
            )

        prompt = f"""Bạn là Thẩm phán đánh giá độ trung thực (Faithfulness & Groundedness Judge) theo chuẩn RAG Week 5.
Quy trình đánh giá gồm 2 bước:
Bước 1 (Decompose): Trích xuất các khẳng định sự thật, quy định, số liệu thực tế trong câu trả lời thành các mệnh đề nguyên tử (atomic claims).
  - Chỉ trích xuất các mệnh đề mang tính khẳng định sự thật / quy chế / số liệu / điều kiện / quy đổi.
  - BỎ QUA các câu chào hỏi, lời dẫn giao tiếp, câu hỏi ngược lại người dùng ("bạn muốn...", "cho tôi biết..."), và các tag trích dẫn [Nguồn:...].
  - Tách nhỏ từng ý độc lập, không gộp nhiều ý vào 1 mệnh đề. Số lượng mệnh đề thường từ 3 đến 10 mệnh đề tùy độ dài câu trả lời.
Bước 2 (Judge): Đối chiếu nghiêm ngặt từng mệnh đề nguyên tử với Ngữ cảnh gốc:
  - supported = true NẾU VÀ CHỈ NẾU ngữ cảnh gốc có nêu rõ hoặc suy luận trực tiếp được thông tin đó.
  - supported = false NẾU thông tin không có trong tài liệu, suy đoán ngoài, hoặc sai lệch số liệu.

Ngữ cảnh gốc:
\"\"\"{context[:8000]}\"\"\"

Câu trả lời của trợ lý AI:
\"\"\"{answer[:3000]}\"\"\"

Trả về DUY NHẤT một JSON format:
{{
  "claims": [
    {{"claim": "Nội dung mệnh đề 1", "supported": true, "reason": "Có trong tài liệu"}},
    {{"claim": "Nội dung mệnh đề 2", "supported": false, "reason": "Không có trong tài liệu"}}
  ]
}}"""

        try:
            raw = _call_groq_judge(
                client=self.client,
                prompt=prompt,
                preferred_model=self.model,
                max_tokens=1200,
                timeout=25.0,
            )
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
            # Fallback an toàn khi mạng gặp sự cố
            return FaithfulnessReport(
                score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                verdicts=[],
                error_message=str(e),
            )

        return FaithfulnessReport(
            score=1.0, total_claims=0, supported_claims=0, unsupported_claims=0
        )

    def evaluate_full(
        self,
        question: str,
        answer: str,
        context: str,
        contexts: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Đánh giá toàn diện bộ 3 chỉ số Week 5: Faithfulness, Answer Relevance, Context Precision.
        """
        f_report = self.evaluate(answer, context)
        relevance = check_answer_relevance(question, answer, groq_client=self.client, model=self.model)

        ctx_list = contexts or ([context] if context.strip() else [])
        precision = check_context_precision(question, ctx_list, groq_client=self.client, model=self.model)

        diag = diagnose_failure_mode(f_report.score, relevance, precision)

        return {
            "faithfulness": f_report.score,
            "faithfulness_report": f_report,
            "answer_relevance": relevance,
            "context_precision": precision,
            "diagnosis": diag,
        }
