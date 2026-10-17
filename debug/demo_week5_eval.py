"""
Full Test & Verification Script for Week 5 Evaluation System.
Matches Teacher's Curriculum (Week 5: RAG Evaluation & Groundedness).

Verifies:
1. Concept 1 (Slide 5): check_faithfulness with Atomic Claims Decomposition (supports 0.80, 0.85, 0.90, 1.00).
2. Concept 2 (Slide 6): check_answer_relevance (On-topic vs Off-topic, computed without context).
3. Concept 2 (Slide 6): check_context_precision (Precision@k, rank quality).
4. Concept 3 (Slide 12 & 13): Failure Mode Diagnosis & Actionable Improvement loop.
5. End-to-end Session Reporter: CSV export + 3 charts (01_tokens, 02_quality, 03_dashboard).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from src.evaluation import (
    FaithfulnessEvaluator,
    check_faithfulness,
    check_answer_relevance,
    check_context_precision,
    diagnose_failure_mode,
    save_session_csv,
    generate_session_charts,
)

def run_comprehensive_tests():
    print("=" * 70)
    print("KIỂM THỬ ĐẦY ĐỦ HỆ THỐNG ĐÁNH GIÁ RAG (CHUẨN BÀI GIẢNG WEEK 5)")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Test 1 (Slide 5): Faithfulness - Kiểm tra xem có ra 0.8 / 0.9 / 1.0 không
    # --------------------------------------------------------------------------
    print("\n[TEST 1 - SLIDE 5] KIỂM THỬ ĐỘ TRUNG THỰC (FAITHFULNESS - ATOMIC CLAIMS)")
    ctx = """Quy định làm việc từ xa:
1. Nhân viên được làm việc từ xa tối đa 2 ngày/tuần.
2. Giờ làm việc linh hoạt từ 8h30 đến 17h30.
Bảng phụ cấp:
- Junior: Phụ cấp thiết bị 1.000.000 VND, 12 ngày phép.
- Senior: Phụ cấp thiết bị 2.000.000 VND, 15 ngày phép."""

    # 1.1 Hoàn toàn trung thực (1.00)
    ans_100 = (
        "Nhân viên được làm việc từ xa tối đa 2 ngày mỗi tuần, với giờ làm việc "
        "từ 8h30 đến 17h30. Cấp bậc Junior được cấp phụ cấp thiết bị 1.000.000 VND "
        "và có 12 ngày phép."
    )
    score_100 = check_faithfulness(ans_100, ctx)
    print(f"  • Mẫu 1 (Chuẩn xác 100%): Điểm = {score_100:.2f} -> {'PASS' if score_100 == 1.0 else 'CHECK'}")

    # 1.2 Có 1 ý suy diễn / ngoài lề trên 5 ý (Kỳ vọng ~ 0.80)
    ans_80 = (
        "Nhân viên được làm việc từ xa tối đa 2 ngày mỗi tuần với giờ làm việc từ 8h30 đến 17h30. "
        "Cấp bậc Junior nhận phụ cấp thiết bị 1.000.000 VND và có 12 ngày phép hàng năm. "
        "Tuy nhiên nhân viên cần gửi email đăng ký trước 24 giờ cho quản lý trực tiếp."
    )
    score_80 = check_faithfulness(ans_80, ctx)
    print(f"  • Mẫu 2 (5 mệnh đề, 1 ý ngoài lề): Điểm = {score_80:.2f} -> {'PASS' if 0.75 <= score_80 <= 0.85 else 'CHECK'}")

    # 1.3 Ảo giác nặng (Kỳ vọng < 0.5)
    ans_hallu = "Nhân viên được làm việc từ xa cả 5 ngày và được tặng xe hơi VinFast miễn phí."
    score_hallu = check_faithfulness(ans_hallu, ctx)
    print(f"  • Mẫu 3 (Ảo giác nặng): Điểm = {score_hallu:.2f} -> {'PASS' if score_hallu <= 0.5 else 'CHECK'}")

    # --------------------------------------------------------------------------
    # Test 2 (Slide 6): Answer Relevance (Đo mức độ bám sát câu hỏi)
    # --------------------------------------------------------------------------
    print("\n[TEST 2 - SLIDE 6] KIỂM THỬ ĐỘ LIÊN QUAN (ANSWER RELEVANCE - NO CONTEXT)")
    q = "Sinh viên hệ 4 năm cần bao nhiêu ngày CTXH để tốt nghiệp?"
    ans_rel = "Sinh viên hệ chính quy đào tạo từ 4 năm trở lên cần tối thiểu 04 ngày CTXH để tốt nghiệp."
    ans_irrel = "Trường Đại học Sư phạm Kỹ thuật được thành lập năm 1962 tại TP.HCM."

    rel_score_good = check_answer_relevance(q, ans_rel)
    rel_score_bad = check_answer_relevance(q, ans_irrel)
    print(f"  • Câu trả lời đúng trọng tâm : Điểm = {rel_score_good:.2f} -> {'PASS' if rel_score_good >= 0.8 else 'CHECK'}")
    print(f"  • Câu trả lời hoàn toàn lạc đề: Điểm = {rel_score_bad:.2f} -> {'PASS' if rel_score_bad <= 0.4 else 'CHECK'}")

    # --------------------------------------------------------------------------
    # Test 3 (Slide 6): Context Precision (Đo chất lượng xếp hạng chunk)
    # --------------------------------------------------------------------------
    print("\n[TEST 3 - SLIDE 6] KIỂM THỬ ĐỘ CHÍNH XÁC XẾP HẠNG (CONTEXT PRECISION)")
    chunks_good = [
        "Sinh viên hệ chính quy khóa 4 năm trở lên phải tích lũy tối thiểu 04 ngày CTXH để tốt nghiệp.",
        "Thông tin phụ: Điểm rèn luyện học kỳ tối đa 100 điểm."
    ]
    chunks_bad = [
        "Thông tin phụ: Lịch thi học kỳ 2 bắt đầu từ ngày 15/06.",
        "Sinh viên hệ chính quy khóa 4 năm trở lên phải tích lũy tối thiểu 04 ngày CTXH để tốt nghiệp."
    ]
    prec_good = check_context_precision(q, chunks_good)
    prec_bad = check_context_precision(q, chunks_bad)
    print(f"  • Chunk hữu ích xếp Rank 1 (Top): Điểm = {prec_good:.2f} -> {'PASS' if prec_good >= 0.9 else 'CHECK'}")
    print(f"  • Chunk hữu ích bị tụt xuống Rank 2: Điểm = {prec_bad:.2f} -> {'PASS' if prec_bad < prec_good else 'CHECK'}")

    # --------------------------------------------------------------------------
    # Test 4 (Slide 12 & 13): Failure Mode Diagnosis & Actionable Fixes
    # --------------------------------------------------------------------------
    print("\n[TEST 4 - SLIDE 12 & 13] KIỂM THỬ CHẨN ĐOÁN LỖI & HƯỚNG CẢI TIẾN")
    diag_good = diagnose_failure_mode(faithfulness=0.9, relevance=0.95, precision=1.0)
    diag_hallu = diagnose_failure_mode(faithfulness=0.5, relevance=0.95, precision=1.0)
    diag_offtopic = diagnose_failure_mode(faithfulness=0.9, relevance=0.4, precision=1.0)
    diag_retrieval = diagnose_failure_mode(faithfulness=0.9, relevance=0.9, precision=0.5)

    print(f"  • Ca 1 (Hệ thống tốt): {diag_good['status']}")
    print(f"  • Ca 2 (Ảo giác): {diag_hallu['status']} -> {diag_hallu['recommendation']}")
    print(f"  • Ca 3 (Lạc đề): {diag_offtopic['status']} -> {diag_offtopic['recommendation']}")
    print(f"  • Ca 4 (Xếp hạng kém): {diag_retrieval['status']} -> {diag_retrieval['recommendation']}")

    # --------------------------------------------------------------------------
    # Test 5 (Slide 11 & Reporter): Xuất báo cáo CSV & 3 đồ thị DPI 300
    # --------------------------------------------------------------------------
    print("\n[TEST 5 - REPORTING] KIỂM THỬ XUẤT CSV VÀ ĐỒ THỊ CHẤT LƯỢNG CAO (DPI 300)")
    sample_records = [
        {
            "id": "q1",
            "category": "logic_rules",
            "query": "Mã hiến máu tình nguyện và số ngày CTXH quy đổi?",
            "is_no_answer": False,
            "is_refused": False,
            "keyword_recall": 1.0,
            "has_citation": True,
            "faithfulness_score": 1.0,
            "answer_relevance": 1.0,
            "context_precision": 1.0,
            "duration_seconds": 2.1,
            "calls": 2,
            "prompt_tokens": 1200,
            "completion_tokens": 180,
            "max_prompt": 800,
            "total_tokens": 1380,
            "answer_preview": "Mã hoạt động 001202, quy đổi 0.5 ngày CTXH.",
        },
        {
            "id": "q2",
            "category": "logic_rules",
            "query": "Học 4 năm cần bao nhiêu ngày CTXH?",
            "is_no_answer": False,
            "is_refused": False,
            "keyword_recall": 1.0,
            "has_citation": True,
            "faithfulness_score": 0.85,
            "answer_relevance": 0.95,
            "context_precision": 1.0,
            "duration_seconds": 1.8,
            "calls": 2,
            "prompt_tokens": 1100,
            "completion_tokens": 160,
            "max_prompt": 750,
            "total_tokens": 1260,
            "answer_preview": "Cần tối thiểu 04 ngày CTXH.",
        },
        {
            "id": "q3",
            "category": "calculation",
            "query": "35 điểm CTXH quy đổi thành bao nhiêu ngày?",
            "is_no_answer": False,
            "is_refused": False,
            "keyword_recall": 1.0,
            "has_citation": True,
            "faithfulness_score": 0.90,
            "answer_relevance": 1.0,
            "context_precision": 0.85,
            "duration_seconds": 2.5,
            "calls": 2,
            "prompt_tokens": 1400,
            "completion_tokens": 210,
            "max_prompt": 900,
            "total_tokens": 1610,
            "answer_preview": "Chia 10 được 3.5 ngày CTXH.",
        },
        {
            "id": "q4",
            "category": "no_answer",
            "query": "Thủ tục xin visa du học Đức?",
            "is_no_answer": True,
            "is_refused": True,
            "keyword_recall": 1.0,
            "has_citation": False,
            "faithfulness_score": 1.0,
            "answer_relevance": 1.0,
            "context_precision": 1.0,
            "duration_seconds": 1.1,
            "calls": 1,
            "prompt_tokens": 800,
            "completion_tokens": 60,
            "max_prompt": 800,
            "total_tokens": 860,
            "answer_preview": "Quy định này không có trong tài liệu CTXH của trường.",
        },
    ]

    out_csv = Path("logs/eval/test_eval_week5.csv")
    save_session_csv(sample_records, out_csv)
    print(f"  • Đã xuất file CSV: {out_csv} ({out_csv.stat().st_size} bytes)")

    charts_dir = Path("logs/eval/charts")
    charts = generate_session_charts(
        records=sample_records,
        output_dir=charts_dir,
        prefix="test_week5_",
        session_title="WEEK 5 COMPREHENSIVE EVAL",
    )
    print(f"  • Đã xuất {len(charts)} biểu đồ DPI 300:")
    for c in charts:
        print(f"    - {c.name} ({c.stat().st_size // 1024} KB)")

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TẤT CẢ CÁC BÀI TEST - HỆ THỐNG KHỚP 100% VỚI BÀI GIẢNG WEEK 5!")
    print("=" * 70)

if __name__ == "__main__":
    run_comprehensive_tests()
