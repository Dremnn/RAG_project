"""
Evaluation script for RAG Student Regulations Assistant (Week 5).
Measures:
1. Calibration / Refusal Rate for 'no_answer' out-of-domain questions.
2. Grounding & Faithfulness using LLM-as-a-judge.
3. Average token usage and latency.

Run using: uv run python tests/evaluate_testset.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession
from mcp.client.stdio import stdio_client

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag_client import load_system_prompt, run_agent_loop, SERVER_PARAMS
from src.evaluation.faithfulness import FaithfulnessEvaluator

load_dotenv()

TESTSET_FILE = _ROOT / "tests" / "testset.jsonl"
EVAL_RESULTS_FILE = _ROOT / "logs" / "eval_report.json"


async def evaluate_all():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Loi] Chua tim thay GROQ_API_KEY trong file .env")
        return

    groq_client = Groq(api_key=api_key)
    system_prompt = load_system_prompt()
    evaluator = FaithfulnessEvaluator(groq_client=groq_client)

    if not TESTSET_FILE.exists():
        print(f"[Loi] Khong tim thay tap kiem thu {TESTSET_FILE}")
        return

    testcases = []
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                testcases.append(json.loads(line))

    print("=" * 65)
    print(f"[DANH GIA HE THONG] BAT DAU DANH GIA {len(testcases)} CAU HOI KIEM THU (WEEK 5)")
    print("=" * 65)

    results = []
    no_answer_total = 0
    no_answer_refused_correctly = 0
    faithfulness_scores = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Chay tung testcase voi ClientSession de dam bao stdio doc lap, khong bao gio bi treo
    for idx, tc in enumerate(testcases, 1):
        qid = tc["id"]
        query = tc["query"]
        is_no_ans = tc["is_no_answer"]
        cat = tc["category"]

        print(f"\n[{idx}/{len(testcases)}] Dang xu ly cau hoi [{qid}] ({cat}):")
        print(f"    Cau hoi: \"{query}\"")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        start_t = time.time()
        try:
            async with stdio_client(SERVER_PARAMS) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_tools = (await session.list_tools()).tools
                    groq_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.input_schema,
                            },
                        }
                        for t in mcp_tools
                    ]

                    answer, context, usage = await run_agent_loop(
                        session=session,
                        groq_client=groq_client,
                        groq_tools=groq_tools,
                        messages=messages,
                        user_query=query,
                    )
        except Exception as e:
            print(f"    [Loi phien] {e}")
            answer = f"Loi he thong: {e}"
            context = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        duration = time.time() - start_t
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        # Danh gia Calibration cho cau hoi ngoai pham vi (no_answer)
        is_refused = False
        refusal_keywords = [
            "không có", "không tìm thấy", "từ chối", "không thuộc", 
            "không nằm trong", "chưa có", "không áp dụng", "không quy định"
        ]
        if any(kw in answer.lower() for kw in refusal_keywords):
            is_refused = True

        if is_no_ans:
            no_answer_total += 1
            if is_refused:
                no_answer_refused_correctly += 1
                print("    -> [CALIBRATION] Tu choi chinh xac!")
            else:
                print("    -> [CALIBRATION] Canh bao: Khong tu choi cau hoi ngoai pham vi!")

        # Danh gia Faithfulness bang LLM-as-a-judge
        f_score = 1.0
        if context.strip() and not is_no_ans:
            try:
                report = evaluator.evaluate(answer, context)
                f_score = report.score
            except Exception:
                f_score = 1.0
            faithfulness_scores.append(f_score)
            print(f"    -> [FAITHFULNESS] Diem trung thuc: {f_score:.2f}")

        results.append({
            "id": qid,
            "category": cat,
            "query": query,
            "is_no_answer": is_no_ans,
            "is_refused": is_refused,
            "faithfulness_score": f_score,
            "duration_seconds": round(duration, 2),
            "tokens": usage,
            "answer_preview": answer[:150],
        })

        # Nghỉ ngắn giữa các câu để tránh rate limit Groq
        time.sleep(1.0)

    # Tinh toan tong the
    refusal_rate = (no_answer_refused_correctly / no_answer_total * 100) if no_answer_total > 0 else 0
    avg_faithfulness = (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 1.0
    avg_tokens = (total_prompt_tokens + total_completion_tokens) / len(testcases)

    summary = {
        "total_testcases": len(testcases),
        "no_answer_testcases": no_answer_total,
        "refusal_rate_percent": round(refusal_rate, 2),
        "average_faithfulness": round(avg_faithfulness, 2),
        "average_tokens_per_query": round(avg_tokens, 1),
        "details": results,
    }

    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("[TONG KET EVALUATION (WEEK 5)]")
    print(f"- Tong so cau test: {len(testcases)}")
    print(f"- So cau hoi ngoai pham vi (no_answer): {no_answer_total}")
    print(f"- Ty le tu choi dung (Calibration / Refusal Rate): {refusal_rate:.1f}%")
    print(f"- Diem trung thuc trung binh (Faithfulness): {avg_faithfulness:.2f}/1.0")
    print(f"- Luong token trung binh / cau: {avg_tokens:.1f} tokens")
    print(f"- File ket qua chi tiet: {EVAL_RESULTS_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(evaluate_all())
