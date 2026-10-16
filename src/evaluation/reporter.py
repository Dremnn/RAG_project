"""
Module Telemetry, Logging & Reporting cho he thong RAG (Week 5 & Week 7).
Cung cap cac ham dung chung (shared utilities) de:
- Ghi log CSV chi tiet tung phien lam viec (Interactive session hoac Benchmark run).
- Tu dong ve va xuat bieu do (Token usage, Quality metrics, Overall dashboard) DPI 300.
"""

import csv
import sys
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


def save_session_csv(records: list[dict[str, Any]], output_file: str | Path) -> Path:
    """
    Xuat danh sach records cua phien chay ra file CSV voi encoding utf-8-sig
    (ho tro mo truc tiep tren Excel Windows khong loi font tieng Viet).
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "id", "category", "query", "is_no_answer", "is_refused",
        "keyword_recall", "has_citation", "faithfulness_score",
        "duration_seconds", "calls", "prompt_tokens", "completion_tokens",
        "max_prompt", "total_tokens", "answer_preview"
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in records:
            usage = r.get("usage", {})
            calls = r.get("calls", usage.get("calls", 1))
            p_tok = r.get("prompt_tokens", usage.get("prompt_tokens", 0))
            c_tok = r.get("completion_tokens", usage.get("completion_tokens", 0))
            m_prompt = r.get("max_prompt", usage.get("max_prompt", p_tok))
            tot_tok = r.get("total_tokens", usage.get("total_tokens", p_tok + c_tok))

            preview = r.get("answer_preview") or r.get("final_answer", "")[:150]
            preview = str(preview).replace("\n", " ").strip()

            writer.writerow([
                r.get("id", ""),
                r.get("category", "general"),
                r.get("query", ""),
                r.get("is_no_answer", False),
                r.get("is_refused", False),
                r.get("keyword_recall", 1.0),
                r.get("has_citation", False),
                r.get("faithfulness_score", 1.0),
                r.get("duration_seconds", 0.0),
                calls,
                p_tok,
                c_tok,
                m_prompt,
                tot_tok,
                preview
            ])

    return path


def generate_session_charts(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    prefix: str = "",
    session_title: str = "PHIÊN CHẠY",
    summary: Optional[dict[str, Any]] = None,
) -> list[Path]:
    """
    Tu dong ve do thi danh gia chat luong cao (DPI 300) cho bat ky phien chay nao.
    Bao gom:
    1. Token Usage Chart (Stacked Prompt + Completion, Line Max Prompt)
    2. Quality Metrics Chart (neu co du lieu Faithfulness / Keyword Recall)
    3. Overall Dashboard 4-in-1 (neu la phien benchmark hoac nhieu cau hoi)
    """
    if not records:
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")  # Chay ngam khong can GUI
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        sys.stderr.write("[Warning] Matplotlib chua duoc cai dat, bo qua buoc ve do thi.\n")
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_charts: list[Path] = []

    ids = [str(r.get("id", f"Q{i+1}")) for i, r in enumerate(records)]
    x = np.arange(len(ids))

    # 1. Do thi Chi phi Token (Usage Tracking)
    try:
        prompts = [r.get("prompt_tokens", r.get("usage", {}).get("prompt_tokens", 0)) for r in records]
        completions = [r.get("completion_tokens", r.get("usage", {}).get("completion_tokens", 0)) for r in records]
        max_prompts = [r.get("max_prompt", r.get("usage", {}).get("max_prompt", 0)) for r in records]

        fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.7), 5), dpi=300)
        ax.bar(x, prompts, 0.55, label="Prompt Tokens", color="#2563eb", alpha=0.85)
        ax.bar(x, completions, 0.55, bottom=prompts, label="Completion Tokens", color="#10b981", alpha=0.85)
        if any(max_prompts):
            ax.plot(x, max_prompts, "r--o", linewidth=1.8, markersize=5, label="Max Prompt / Step")

        ax.set_title(f"CHI PHÍ TOKEN CHI TIẾT - {session_title.upper()}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Mã câu hỏi / Lượt tương tác", fontsize=10, labelpad=6)
        ax.set_ylabel("Số lượng Tokens", fontsize=10, labelpad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=9, fontweight="bold", rotation=30 if len(ids) > 8 else 0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white")
        plt.tight_layout()

        p1 = out_dir / f"{prefix}01_token_usage.png"
        fig.savefig(p1)
        plt.close(fig)
        generated_charts.append(p1)
    except Exception as e:
        sys.stderr.write(f"[Warning] Loi khi ve bieu do Token: {e}\n")

    # 2. Do thi Chat luong (neu co it nhat mot chi so danh gia)
    has_eval_metrics = any("faithfulness_score" in r or "keyword_recall" in r for r in records)
    if has_eval_metrics:
        try:
            faith = [float(r.get("faithfulness_score", 1.0)) * 100 for r in records]
            recall = [float(r.get("keyword_recall", 1.0)) * 100 for r in records]
            width = 0.35

            fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.7), 5), dpi=300)
            ax.bar(x - width/2, faith, width, label="Faithfulness (%)", color="#059669", alpha=0.9)
            ax.bar(x + width/2, recall, width, label="Keyword Recall (%)", color="#0284c7", alpha=0.9)
            ax.axhline(80, color="#ef4444", linestyle=":", linewidth=1.5, label="Ngưỡng tin cậy (80%)")

            ax.set_title(f"CHỈ SỐ CHẤT LƯỢNG RAG - {session_title.upper()}", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("Mã câu hỏi / Lượt tương tác", fontsize=10, labelpad=6)
            ax.set_ylabel("Tỷ lệ (%)", fontsize=10, labelpad=6)
            ax.set_xticks(x)
            ax.set_xticklabels(ids, fontsize=9, fontweight="bold", rotation=30 if len(ids) > 8 else 0)
            ax.set_ylim(0, 115)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(frameon=True, facecolor="white")
            plt.tight_layout()

            p2 = out_dir / f"{prefix}02_quality_metrics.png"
            fig.savefig(p2)
            plt.close(fig)
            generated_charts.append(p2)
        except Exception as e:
            sys.stderr.write(f"[Warning] Loi khi ve bieu do Chat luong: {e}\n")

    # 3. Dashboard Tong hop 4-in-1 (khi co summary hoac co >= 4 cau hoi)
    if summary or len(records) >= 4:
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

            # Subplot 1: Categories
            cats = {}
            for r in records:
                c = r.get("category", "general")
                cats[c] = cats.get(c, 0) + 1
            ax1.pie(list(cats.values()), labels=list(cats.keys()), autopct="%1.0f%%", startangle=140,
                    colors=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"])
            ax1.set_title("1. Phân bố thể loại", fontsize=10, fontweight="bold")

            # Subplot 2: Calibration rate / Refusal
            ref_pct = summary.get("refusal_rate_percent", 100.0) if summary else 100.0
            no_ans = summary.get("no_answer_testcases", sum(1 for r in records if r.get("is_no_answer"))) if summary else 1
            if no_ans > 0:
                ref_cnt = int(round(no_ans * ref_pct / 100))
                fail_cnt = no_ans - ref_cnt
                ax2.pie([max(ref_cnt, 1), max(fail_cnt, 0)], labels=["Từ chối đúng", "Chưa từ chối"],
                        autopct="%1.0f%%", startangle=90, colors=["#10b981", "#ef4444"])
            else:
                ax2.text(0.5, 0.5, "100% In-Scope", ha="center", va="center", fontsize=11, fontweight="bold")
            ax2.set_title(f"2. Calibration (Refusal: {ref_pct}%)", fontsize=10, fontweight="bold")

            # Subplot 3: Total token per question
            totals = [r.get("total_tokens", r.get("usage", {}).get("total_tokens", 0)) for r in records]
            ax3.bar(ids, totals, color="#3b82f6", alpha=0.85)
            ax3.set_title("3. Tổng Token / Lượt", fontsize=10, fontweight="bold")
            ax3.tick_params(axis="x", rotation=45 if len(ids) > 6 else 0)
            ax3.grid(axis="y", linestyle="--", alpha=0.5)

            # Subplot 4: Summary Info Box
            ax4.axis("off")
            total_cases = len(records)
            avg_faith = (sum(float(r.get("faithfulness_score", 1.0)) for r in records) / total_cases) if total_cases else 1.0
            avg_recall = (sum(float(r.get("keyword_recall", 1.0)) for r in records) / total_cases) if total_cases else 1.0
            avg_tok = (sum(totals) / total_cases) if total_cases else 0

            info = (
                f"4. TỔNG HỢP {session_title.upper()}\n"
                "────────────────────────────────────────────\n"
                f"• Tổng số lượt truy vấn: {total_cases} câu\n"
                f"• Tỷ lệ từ chối đúng   : {ref_pct:.1f}%\n"
                f"• Độ trung thực (Faith): {avg_faith:.2f} / 1.00\n"
                f"• Độ phủ từ khóa        : {avg_recall*100:.1f}%\n"
                f"• Token trung bình      : {avg_tok:,.1f} tok/câu\n"
                "────────────────────────────────────────────\n"
                "Trạng thái: HOÀN TẤT PHIÊN LÀM VIỆC"
            )
            ax4.text(0.05, 0.5, info, fontsize=9.5, family="monospace",
                     verticalalignment="center", bbox=dict(boxstyle="round,pad=1", facecolor="#f8fafc", edgecolor="#cbd5e1"))

            plt.suptitle(f"BẢNG TỔNG HỢP {session_title.upper()}", fontsize=13, fontweight="bold")
            plt.tight_layout()

            p3 = out_dir / f"{prefix}03_overall_dashboard.png"
            fig.savefig(p3)
            plt.close(fig)
            generated_charts.append(p3)
        except Exception as e:
            sys.stderr.write(f"[Warning] Loi khi ve Dashboard: {e}\n")

    return generated_charts


def save_and_plot_session(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    session_id: str,
    session_title: str = "Phiên tương tác",
    summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Ham tien ich tong hop: Vua xuat file CSV vua ve toan bo do thi cho mot phien chay.
    Tra ve duong dan file CSV va danh sach cac file do thi da sinh.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"session_{session_id}.csv"
    save_session_csv(records, csv_path)

    charts = generate_session_charts(
        records=records,
        output_dir=out_dir / "charts",
        prefix=f"{session_id}_",
        session_title=f"{session_title} ({session_id})",
        summary=summary,
    )

    return {
        "session_id": session_id,
        "csv_path": csv_path,
        "charts": charts,
        "total_records": len(records),
    }
