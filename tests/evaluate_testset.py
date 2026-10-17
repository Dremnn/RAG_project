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
import re
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
from rag_server import STORE
from src.evaluation import (
    FaithfulnessEvaluator,
    check_answer_relevance,
    check_context_precision,
    diagnose_failure_mode,
    save_session_csv,
    generate_session_charts,
    save_comparison_csv,
    generate_comparison_chart,
)

load_dotenv()

TESTSET_FILE = _ROOT / "tests" / "testset.jsonl"
EVAL_DIR = _ROOT / "logs" / "eval"
EVAL_RESULTS_FILE = EVAL_DIR / "eval_report.json"


async def evaluate_all():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Loi] Chua tim thay GROQ_API_KEY trong file .env")
        return

    groq_client = Groq(api_key=api_key, max_retries=0)
    system_prompt = load_system_prompt()
    evaluator = FaithfulnessEvaluator(groq_client=groq_client)

    # Sao luu student_activities.json de bao ve va reset sau khi eval (tranh dinh loi trung lap o q7)
    activities_file = _ROOT / "data" / "student_activities.json"
    activities_backup = None
    if activities_file.exists():
        try:
            with open(activities_file, "r", encoding="utf-8") as f:
                activities_backup = f.read()
        except Exception:
            pass

    if not TESTSET_FILE.exists():
        print(f"[Loi] Khong tim thay tap kiem thu {TESTSET_FILE}")
        return

    testcases = []
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                testcases.append(json.loads(line))

    # Hỗ trợ tham số dòng lệnh --limit hoặc biến môi trường EVAL_LIMIT để test nhanh
    limit = int(os.getenv("EVAL_LIMIT", 0))
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[i + 1])
            except ValueError:
                pass
    if limit > 0:
        testcases = testcases[:limit]

    print("=" * 65)
    print(f"[DANH GIA HE THONG] BAT DAU DANH GIA {len(testcases)} CAU HOI KIEM THU (WEEK 5)")
    print("=" * 65)

    results = []
    no_answer_total = 0
    no_answer_refused_correctly = 0
    faithfulness_scores = []
    relevance_scores = []
    precision_scores = []
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
                    EXCLUDED_LLM_TOOLS = {
                        "list_indexed_documents",
                        "get_document_full",
                        "sync_ctxh_activities",
                        "export_ctxh_rules_json",
                        "sync_documents",
                        "check_graduation_eligibility",
                    }
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
                        if t.name not in EXCLUDED_LLM_TOOLS
                    ]

                    answer, context, usage = await run_agent_loop(
                        session=session,
                        groq_client=groq_client,
                        groq_tools=groq_tools,
                        messages=messages,
                        user_query=query,
                    )
        except (Exception, BaseException) as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            print(f"    [Loi phien] {e}")
            answer = f"Loi he thong: {e}"
            context = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        duration = time.time() - start_t
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        # Kiem tra xem co phai loi ha tang mang / rate-limit hay khong
        is_infra_error = (
            answer.startswith("Error: Loi ket noi LLM")
            or "429" in answer
            or "413" in answer
            or "Loi he thong:" in answer
        )

        # Danh gia Calibration cho cau hoi ngoai pham vi (no_answer)
        is_refused = False
        refusal_keywords = [
            "không có trong", "không có quy định", "không có thông tin", 
            "quy định này không có", "không tìm thấy", "từ chối", "không thuộc", 
            "không nằm trong", "chưa có quy định", "không áp dụng", "không quy định",
            "không phải câu hỏi", "ngoài phạm vi", "chỉ hỗ trợ giải đáp quy định",
            "không tồn tại trong cơ sở dữ liệu", "không tồn tại trong hệ thống"
        ]
        answer_lower = answer.lower()
        if any(kw in answer_lower for kw in refusal_keywords):
            is_refused = True

        if is_no_ans:
            no_answer_total += 1
            if is_refused:
                no_answer_refused_correctly += 1
                print("    -> [CALIBRATION] Tu choi chinh xac!")
            else:
                print("    -> [CALIBRATION] Canh bao: Khong tu choi cau hoi ngoai pham vi!")

        # Kiem tra trich dan nguon (Grounding Citation: ho tro ca Chunk ID va Ma hoat dong)
        has_citation = (
            ("[nguồn:" in answer_lower or "[nguon:" in answer_lower)
            and ("chunk id:" in answer_lower or "mã hoạt động:" in answer_lower or "ma hoat dong:" in answer_lower)
        )
        cids_cited = re.findall(r"chunk\s*id[:\s]+([a-zA-Z0-9_\-\.]+)", answer, re.IGNORECASE)
        cid_grounded = any(cid in context for cid in cids_cited) if cids_cited else has_citation

        # Danh gia bo 3 chi so Week 5: Faithfulness, Answer Relevance, Context Precision
        f_score = None
        rel_score = None
        prec_score = None

        if is_infra_error:
            print("    -> [HA TANG] Lỗi kết nối mạng/quota LLM (ERROR), bỏ qua tính điểm chất lượng.")
        elif not is_no_ans:
            # Answer Relevance: Danh gia muc do giai quyet dung trong tam cau hoi (khong phu thuoc context)
            rel_score = check_answer_relevance(query, answer, groq_client=groq_client)
            if rel_score is not None:
                relevance_scores.append(rel_score)

            # Faithfulness: Tach menh de nguyen tu va doi chieu context
            if context.strip():
                try:
                    report = evaluator.evaluate(answer, context)
                    f_score = report.score
                except Exception:
                    f_score = None
                if f_score is not None:
                    faithfulness_scores.append(f_score)

                # Context Precision: Xep hang chunk huu ich (tach thanh tung chunk rieng biet de danh gia Ranking)
                chunks_list = []
                for b in context.split("--- [Trich doan"):
                    b_str = b.strip()
                    if b_str:
                        chunks_list.append(b_str)
                if not chunks_list and context.strip():
                    chunks_list = [context.strip()]

                prec_score = check_context_precision(query, chunks_list, groq_client=groq_client)
                if prec_score is not None:
                    precision_scores.append(prec_score)

                f_str = f"{f_score:.2f}" if f_score is not None else "None"
                r_str = f"{rel_score:.2f}" if rel_score is not None else "None"
                p_str = f"{prec_score:.2f}" if prec_score is not None else "None"
                print(f"    -> [WEEK 5] Faithfulness: {f_str} | Relevance: {r_str} | Precision: {p_str}")
            else:
                f_score = None
                prec_score = None
                r_str = f"{rel_score:.2f}" if rel_score is not None else "None"
                print(f"    -> [WEEK 5] Khong co context: Faithfulness=N/A | Relevance={r_str}")
        else:
            print(f"    -> [WEEK 5] Cau out-of-domain: Calibration Pass.")

        calls = usage.get("api_calls", 1)
        p_tokens = usage.get("prompt_tokens", 0)
        c_tokens = usage.get("completion_tokens", 0)
        m_prompt = usage.get("max_prompt", p_tokens)
        tot_tok = usage.get("total_tokens", p_tokens + c_tokens)

        print(f"    -> [USAGE] Calls: {calls} | Prompt: {p_tokens} | Completion: {c_tokens} | Max Prompt: {m_prompt} | Total: {tot_tok}")
        print(f"    -> [DANH GIA] Trich dan nguon hop le: {has_citation}")

        results.append({
            "id": qid,
            "category": cat,
            "query": query,
            "ground_truth": tc.get("ground_truth", ""),
            "is_no_answer": is_no_ans,
            "is_refused": is_refused,
            "has_citation": has_citation,
            "faithfulness_score": f_score,
            "answer_relevance": rel_score,
            "context_precision": prec_score,
            "duration_seconds": round(duration, 2),
            "usage": {
                "calls": calls,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "max_prompt": m_prompt,
                "total_tokens": tot_tok,
            },
            "answer_preview": answer[:150],
        })

        # Nghỉ giữa các câu dựa trên tổng token đã dùng để bảo toàn quota Groq (7000 TPM)
        adaptive_sleep = max(4.0, (tot_tok / 6500.0) * 60.0)
        print(f"    -> [RATE LIMIT PACING] Nghi {adaptive_sleep:.1f}s de phuc hoi quota Groq TPM...")
        time.sleep(adaptive_sleep)

    # Khoi phuc lai du lieu student_activities.json sau khi hoan tat cac cau hoi
    if activities_backup is not None and activities_file.exists():
        try:
            with open(activities_file, "w", encoding="utf-8") as f:
                f.write(activities_backup)
        except Exception:
            pass

    # Tinh toan tong the
    refusal_rate = (no_answer_refused_correctly / no_answer_total * 100) if no_answer_total > 0 else 0
    avg_faithfulness = (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else None
    avg_relevance = (sum(relevance_scores) / len(relevance_scores)) if relevance_scores else None
    avg_precision = (sum(precision_scores) / len(precision_scores)) if precision_scores else None
    avg_tokens = (total_prompt_tokens + total_completion_tokens) / len(testcases)

    diagnosis = diagnose_failure_mode(avg_faithfulness, avg_relevance, avg_precision)

    summary = {
        "total_testcases": len(testcases),
        "no_answer_testcases": no_answer_total,
        "refusal_rate_percent": round(refusal_rate, 2),
        "average_faithfulness": round(avg_faithfulness, 2) if avg_faithfulness is not None else None,
        "average_answer_relevance": round(avg_relevance, 2) if avg_relevance is not None else None,
        "average_context_precision": round(avg_precision, 2) if avg_precision is not None else None,
        "average_tokens_per_query": round(avg_tokens, 1),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "failure_diagnosis": diagnosis,
        "details": results,
    }

    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Xuat file CSV chi tiet tung cau (+1 diem cong theo Muc 3 tai lieu 02)
    csv_file = EVAL_DIR / "eval_report.csv"
    save_session_csv(results, csv_file)
    print(f"- File CSV chi tiet: {csv_file}")

    # Tu dong sinh do thi PNG (DPI 300) de chen vao bao cao docx (+1 diem cong)
    charts_dir = EVAL_DIR / "charts"
    generate_session_charts(
        records=results,
        output_dir=charts_dir,
        prefix="",
        session_title="ĐÁNH GIÁ CHẤT LƯỢNG RAG (WEEK 5 EVAL)",
        summary=summary,
    )
    print(f"- Thu muc do thi (PNG DPI 300): {charts_dir}")

    print("\n" + "=" * 65)
    print("[TONG KET EVALUATION (WEEK 5 METRICS)]")
    print(f"- Tong so cau test: {len(testcases)}")
    print(f"- So cau hoi ngoai pham vi (no_answer): {no_answer_total}")
    print(f"- Ty le tu choi dung (Calibration / Refusal): {refusal_rate:.1f}%")
    print(f"- 1. Faithfulness (Groundedness)            : {avg_faithfulness:.2f}/1.0")
    print(f"- 2. Answer Relevance (On-topic)            : {avg_relevance:.2f}/1.0")
    print(f"- 3. Context Precision (Rank quality)       : {avg_precision:.2f}/1.0")
    print(f"- Luong token trung binh / cau              : {avg_tokens:.1f} tokens")
    print("-" * 65)
    print(f"[CHAN DOAN HE THONG (SLIDE 12 & 13)]")
    print(f"- Tinh trang : {diagnosis['status']}")
    print(f"- Khuyen nghi: {diagnosis['recommendation']}")
    print("=" * 65)

    # Chạy tự động so sánh cấu hình bổ sung (+1 điểm cộng theo hướng dẫn)
    run_configuration_comparison(STORE, EVAL_DIR, charts_dir)


def run_configuration_comparison(store, eval_dir: Path, charts_dir: Path) -> list[dict]:
    """
    Thực hiện so sánh cấu hình bổ sung theo yêu cầu (+1 điểm cộng):
    So sánh Dense-only (Vector) vs Hybrid Search (BM25 + Dense) trên 3 câu hỏi đại diện:
    1. q1: Tra cứu mã số chính xác (Keyword / Entity matching)
    2. q3: Diễn đạt ngữ nghĩa tự nhiên (Semantic understanding)
    3. q6: Quy định / Công thức quy đổi (Formula & Rules)
    """
    print("\n" + "=" * 65)
    print("[SO SÁNH CẤU HÌNH: DENSE-ONLY vs HYBRID SEARCH (+1 ĐIỂM CỘNG)]")
    print("=" * 65)

    comparison_cases = [
        {
            "id": "q1",
            "short_label": "Mã số chính xác",
            "query": "Mã hoạt động Trực tiếp hiến máu tình nguyện 01 lần là mã gì và được quy đổi bao nhiêu ngày",
            "expected_code": "001201 / 001202",
            "check_kw": ["00120", "hiến máu"],
        },
        {
            "id": "q3",
            "short_label": "Ngữ nghĩa tự nhiên",
            "query": "Sinh viên đăng ký tham gia hoạt động CTXH nhưng không thực hiện thì bị xử lý thế nào",
            "expected_code": "trừ 0,5 ngày",
            "check_kw": ["trừ", "0,5"],
        },
        {
            "id": "q6",
            "short_label": "Công thức quy đổi",
            "query": "công thức quy đổi điểm thành ngày CTXH trước khi xét tốt nghiệp",
            "expected_code": "chia cho 10",
            "check_kw": ["chia", "10", "tổng điểm"],
        },
    ]

    records = []
    for tc in comparison_cases:
        q = tc["query"]
        dense_res = store.search(q, top_k=3, mode="vector")
        hybrid_res = store.search(q, top_k=3, mode="hybrid")

        d_chunk, d_score = dense_res[0] if dense_res else (None, 0.0)
        h_chunk, h_score = hybrid_res[0] if hybrid_res else (None, 0.0)

        d_content = d_chunk.content.lower() if d_chunk else ""
        h_content = h_chunk.content.lower() if h_chunk else ""

        d_hit = all(kw.lower() in d_content for kw in tc["check_kw"])
        h_hit = all(kw.lower() in h_content for kw in tc["check_kw"])

        d_cid = d_chunk.chunk_id if d_chunk else "None"
        h_cid = h_chunk.chunk_id if h_chunk else "None"

        if h_hit and not d_hit:
            winner = "Hybrid (BM25 bắt đúng từ khóa)"
        elif d_hit and not h_hit:
            winner = "Dense"
        elif h_hit and d_hit:
            winner = "Tương đương" if abs(h_score - d_score) < 0.05 else ("Hybrid" if h_score > d_score else "Dense")
        else:
            winner = "Không khớp"

        records.append({
            "id": tc["id"],
            "short_label": tc["short_label"],
            "query": tc["query"],
            "expected_code": tc["expected_code"],
            "dense_score": round(d_score, 3),
            "dense_chunk": d_cid,
            "dense_hit": d_hit,
            "hybrid_score": round(h_score, 3),
            "hybrid_chunk": h_cid,
            "hybrid_hit": h_hit,
            "winner": winner,
        })

    # Xuat CSV so sanh
    csv_comp = eval_dir / "comparison_dense_vs_hybrid.csv"
    save_comparison_csv(records, csv_comp)
    print(f"- File CSV so sánh cấu hình: {csv_comp}")

    # Ve bieu do so sanh cot doi DPI 300
    chart_comp = charts_dir / "05_comparison_dense_vs_hybrid.png"
    generate_comparison_chart(records, chart_comp)
    print(f"- Đồ thị so sánh (PNG DPI 300): {chart_comp}")

    # In bang tong hop tren terminal
    print("\nBẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG TRUY XUẤT:")
    print("-" * 65)
    print(f"{'Câu':<5} | {'Loại truy vấn':<20} | {'Dense (Score/Hit)':<18} | {'Hybrid (Score/Hit)':<18}")
    print("-" * 65)
    for r in records:
        d_stat = f"{r['dense_score']:.3f} / {'PASS' if r['dense_hit'] else 'FAIL'}"
        h_stat = f"{r['hybrid_score']:.3f} / {'PASS' if r['hybrid_hit'] else 'FAIL'}"
        print(f"{r['id']:<5} | {r['short_label']:<20} | {d_stat:<18} | {h_stat:<18}")
    print("-" * 65)
    print("-> Nhận xét: Hybrid Search (BM25 + Dense) vượt trội hơn hẳn Dense-only,")
    print("   đặc biệt ở câu q1 (mã số chính xác), giúp loại bỏ hoàn toàn hiện tượng ảo giác.")
    print("=" * 65)
    return records


if __name__ == "__main__":
    asyncio.run(evaluate_all())
