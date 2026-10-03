"""
Demo script to test Groundedness & Faithfulness Evaluation (Week 5).
Run using: uv run python demo_evaluation.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.evaluation import FaithfulnessEvaluator

load_dotenv()


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Lỗi: Chưa có GROQ_API_KEY trong file .env")
        return

    groq_client = Groq(api_key=api_key)
    evaluator = FaithfulnessEvaluator(groq_client=groq_client)

    # 1. Real context from sample_company_policy.docx
    context = """
    Quy định làm việc từ xa:
    1. Nhân viên được làm việc từ xa tối đa 2 ngày/tuần.
    2. Giờ làm việc linh hoạt từ 8h30 đến 17h30.
    Bảng phụ cấp:
    - Junior: Phụ cấp thiết bị 1.000.000 VND, 12 ngày phép.
    - Senior: Phụ cấp thiết bị 2.000.000 VND, 15 ngày phép.
    """

    print("=" * 65)
    print("🧪 KIỂM THỬ THẨM ĐỊNH FAITHFULNESS (LLM-AS-A-JUDGE)")
    print("=" * 65)

    # Case 1: Faithful Answer
    print("\n[Trường hợp 1]: Câu trả lời hoàn toàn trung thực dựa trên tài liệu")
    faithful_answer = (
        "Nhân viên được phép làm việc từ xa tối đa 2 ngày một tuần. "
        "Mức phụ cấp thiết bị dành cho nhân viên cấp bậc Junior là 1.000.000 VND."
    )
    print(f"Câu trả lời: \"{faithful_answer}\"")
    print("Đang thẩm định...")
    report1 = evaluator.evaluate(faithful_answer, context)
    print(report1.format_terminal())

    # Case 2: Hallucinated Answer (Bịa đặt)
    print("\n[Trường hợp 2]: Câu trả lời chứa thông tin bịa đặt (Ảo giác)")
    hallucinated_answer = (
        "Nhân viên được làm việc từ xa toàn bộ 5 ngày mỗi tuần mà không cần xin phép. "
        "Ngoài ra công ty tặng thêm mỗi nhân viên một chiếc xe hơi VinFast miễn phí."
    )
    print(f"Câu trả lời: \"{hallucinated_answer}\"")
    print("Đang thẩm định...")
    report2 = evaluator.evaluate(hallucinated_answer, context)
    print(report2.format_terminal())


if __name__ == "__main__":
    main()
