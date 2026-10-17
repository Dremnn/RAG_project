"""
Terminal RAG MCP Client with ReAct Agent Loop (Week 6 & Week 7).
Connects to rag_server.py over stdio and interacts with user in the terminal.
Designed for: Tro ly Quy che & Thu tuc Sinh vien.
Run using: uv run python rag_client.py

Kiến trúc mã nguồn được phân tách làm 2 phần rõ rệt:
- PHẦN 1: CORE REACT AGENT LOOP & MCP INTERACTION (Suy luận, gọi tools, xử lý LLM Groq)
- PHẦN 2: SESSION TELEMETRY & REPORTING (Ủy thác qua SessionTracker: ghi log jsonl, xuất CSV, vẽ biểu đồ)
"""

import asyncio
import json
import os
import sys
import re   
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.evaluation import (
    FaithfulnessEvaluator,
    check_answer_relevance,
    SessionTracker,
)


# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

HERE = Path(__file__).resolve().parent
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "0") == "1"
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"] if ENABLE_FALLBACK else []
MAX_STEPS = 7
LOGS_DIR = HERE / "logs" / "app"
PROMPT_PATH = HERE / "prompts" / "system_agent.txt"

# Prefer virtualenv python if available, fallback to sys.executable
VENV_PYTHON = HERE.parent / ".venv" / "Scripts" / "python.exe"
PYTHON_CMD = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

SERVER_PARAMS = StdioServerParameters(
    command=PYTHON_CMD,
    args=[str(HERE / "rag_server.py")],
    env=dict(os.environ),
)


# CORE REACT AGENT LOOP & MCP INTERACTION

def load_system_prompt() -> str:
    """Load system prompt from external file (Week 2 clean architecture)."""
    if PROMPT_PATH.exists():
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return (
        "Ban la Tro ly AI Quy che & Thu tuc Sinh vien. "
        "Luon tra cuu van ban quy che qua search_ctxh_rules va trich dan nguon [Nguon: ... | Chunk ID: ...]."
    )


def call_groq_with_retry(
    groq_client: Groq,
    messages: list[dict],
    tools: list[dict],
    tool_choice: str = "auto",
    max_retries: int = 5,
    timeout: float = 30.0,
) -> tuple[any, dict]:
    """Call Groq API with exponential backoff retry, rate limit handling, and model fallback."""
    enable_fallback = os.getenv("ENABLE_FALLBACK", "0") == "1"
    candidate_models = [MODEL]
    if enable_fallback:
        candidate_models += [m for m in FALLBACK_MODELS if m != MODEL]
    last_err = None

    for model_name in candidate_models:
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=750,
                    timeout=timeout,
                )
                usage_meta = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                    "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
                }
                return response, usage_meta
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                print(f"[Canh bao] Model {model_name} gap su co (Thu lan {attempt}/{max_retries}): {e}")

                # 413: Payload too large - request vuot nguong kich thuoc, retry cung se loi -> ngat ngay
                if "413" in err_str or "too large" in err_str or "payload" in err_str:
                    raise RuntimeError(f"Request qua lon vuot nguong Groq (413 Payload Too Large): {e}")

                # Neu cham han muc token theo ngay (TPD), fallback sang model tiep theo neu bat fallback
                if "tokens per day" in err_str or "tpd" in err_str:
                    print(f"[Fallback] Model {model_name} vuot han muc TPD. Chuyen sang model tiep theo...")
                    break

                if attempt < max_retries:
                    if "rate_limit" in err_str or "429" in err_str:
                        m = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)", str(e), re.IGNORECASE)
                        if m:
                            val = float(m.group(1))
                            unit = m.group(2).lower()
                            wait = (val / 1000.0) if unit == "ms" else val
                        else:
                            wait = 15.0
                        wait_time = min(wait + 1.0, 65.0)
                        print(f"[Rate Limit 429] Cho {wait_time:.1f} giay theo yeu cau cua Groq (Thu lan {attempt}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        time.sleep(delay)
                        delay *= 2

    raise RuntimeError(f"Goi Groq API that bai tren toan bo models: {last_err}")


async def run_agent_loop(
    session: ClientSession,
    groq_client: Groq,
    groq_tools: list[dict],
    messages: list[dict],
    user_query: str,
    tracker: SessionTracker | None = None,
) -> tuple[str, str, dict]:
    """
    Custom ReAct Agent Loop (Week 6 & Week 7).
    - MAX_STEPS = 7
    - Strict Grounding: First step enforces tool call
    - Ghi nhận trajectory và đẩy trace vào tracker.
    """
    accumulated_context = []
    trajectory = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0, "max_prompt": 0}
    registered_mssv = None
    verified_mssv = set()

    for step in range(MAX_STEPS):
        # Tu dong quyet dinh tool_choice: 'auto' cac buoc dau de cho phep tu choi ngay; 'none' o buoc cuoi de ep tong hop
        tool_choice = "none" if step == MAX_STEPS - 1 else "auto"

        try:
            response, usage = call_groq_with_retry(
                groq_client=groq_client,
                messages=messages,
                tools=groq_tools,
                tool_choice=tool_choice,
            )
        except Exception as e:
            err_msg = f"Error: Loi ket noi LLM: {e}"
            print(f"[Loi] {err_msg}")
            if tracker:
                tracker.log_agent_run({
                    "timestamp": datetime.now().isoformat(),
                    "query": user_query,
                    "status": "ERROR",
                    "error": str(e),
                    "steps": step,
                    "trajectory": trajectory,
                    "final_answer": err_msg,
                    "usage": total_tokens,
                })
            return err_msg, "\n\n".join(accumulated_context), total_tokens

        total_tokens["prompt_tokens"] += usage["prompt_tokens"]
        total_tokens["completion_tokens"] += usage["completion_tokens"]
        total_tokens["total_tokens"] += usage["total_tokens"]
        total_tokens["api_calls"] += 1
        total_tokens["max_prompt"] = max(total_tokens["max_prompt"], usage.get("prompt_tokens", 0))

        response_message = response.choices[0].message

        # Kiem tra xem LLM co yeu cau goi Tool khong
        if response_message.tool_calls:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_args = {}

                print(f"\n[Hanh dong - Buoc {step + 1}] Goi cong cu: '{tool_name}'")
                if tool_name in ["search_ctxh_rules", "search_regulations", "search_documents"]:
                    print(f"    -> Tu khoa: \"{tool_args.get('query')}\"")
                    if tool_args.get("year"):
                        print(f"    -> Nam (year): {tool_args.get('year')}")
                elif tool_name == "get_activity":
                    print(f"    -> [TRA CUU QUY CHE] Ma/Tu khoa: {tool_args.get('code_or_name')}")
                elif tool_name in ["check_student_exists", "check_graduation_eligibility"]:
                    print(f"    -> [THAM DINH HO SO] MSSV: {tool_args.get('mssv')}")
                elif tool_name in ["register_ctxh_activity", "submit_academic_request"]:
                    print(f"    -> [GHI DU LIEU] MSSV: {tool_args.get('mssv')} | Ma hoat dong: {tool_args.get('activity_code', '')}")
                elif tool_name in ["get_ctxh_status", "get_request_status"]:
                    print(f"    -> [KIEM CHUNG] MSSV: {tool_args.get('mssv', tool_args.get('request_id', ''))}")

                # Goi Tool tren MCP Server qua stdio an toan (khong crash)
                try:
                    tool_result = await session.call_tool(tool_name, arguments=tool_args)
                    result_text = tool_result.content[0].text if tool_result.content else ""
                except Exception as e:
                    result_text = f"Error: Loi khi thuc thi cong cu '{tool_name}': {e}"

                print(f"    -> Nhan phan hoi: {len(result_text)} ky tu.")
                accumulated_context.append(result_text)

                if tool_name == "register_ctxh_activity" and not result_text.startswith("Error"):
                    registered_mssv = str(tool_args.get("mssv", "")).strip()
                elif tool_name == "get_ctxh_status":
                    verified_mssv.add(str(tool_args.get("mssv", "")).strip())

                trajectory.append({
                    "step": step + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": result_text[:500] if len(result_text) > 500 else result_text,
                })

                # Gui ket qua quan sat (Observation) lai cho LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

            continue

        # [VERIFY GUARD] Kiem tra xem co hoat dong ghi nao chua duoc verify qua get_ctxh_status khong
        if registered_mssv and registered_mssv not in verified_mssv and step < MAX_STEPS - 1:
            print(f"\n    -> [VERIFY GUARD] Phat hien dang ky chua kiem chung. Tu dong thuc thi get_ctxh_status cho MSSV {registered_mssv}...")
            try:
                v_res = await session.call_tool("get_ctxh_status", arguments={"mssv": registered_mssv})
                v_text = v_res.content[0].text if v_res.content else ""
            except Exception as e:
                v_text = f"Error: Loi khi tu dong verify: {e}"

            verified_mssv.add(registered_mssv)
            accumulated_context.append(v_text)
            trajectory.append({
                "step": step + 1,
                "tool": "get_ctxh_status",
                "args": {"mssv": registered_mssv},
                "result_preview": v_text[:500] if len(v_text) > 500 else v_text,
            })
            messages.append({
                "role": "user",
                "content": f"[XÁC THỰC BẮT BUỘC TỪ HỆ THỐNG]: Đã tự động kiểm chứng qua get_ctxh_status:\n{v_text}\nHãy hoàn tất câu trả lời chính thức cho sinh viên, nêu rõ mã đăng ký, trạng thái CONFIRMED và tiến độ tích lũy.",
            })
            continue

        # Neu khong con tool_calls -> LLM da hoan tat suy luan va tra ve cau tra loi cuoi cung
        final_answer = response_message.content or ""
        messages.append({"role": "assistant", "content": final_answer})

        # Ghi log jsonl qua SessionTracker (Phần 2)
        if tracker:
            tracker.log_agent_run({
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "status": "SUCCESS",
                "steps": len(trajectory),
                "trajectory": trajectory,
                "final_answer": final_answer,
                "usage": total_tokens,
            })

        return final_answer, "\n\n".join(accumulated_context), total_tokens

    # Qua so buoc cho phep (Max steps reached)
    warning_msg = (
        f"[Thong bao] Da vuot qua gioi han {MAX_STEPS} buoc xu ly cho phep. "
        "Vui long cung cap yeu cau cu the hon de tro ly phuc vu tot hon."
    )
    if tracker:
        tracker.log_agent_run({
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "steps": MAX_STEPS,
            "trajectory": trajectory,
            "final_answer": warning_msg,
            "usage": total_tokens,
            "status": "MAX_STEPS_REACHED",
        })

    return warning_msg, "\n\n".join(accumulated_context), total_tokens


# TERMINAL CLI & TELEMETRY INTEGRATION


async def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Loi] Chua tim thay GROQ_API_KEY trong file .env")
        return

    groq_client = Groq(api_key=api_key, max_retries=0)
    system_prompt = load_system_prompt()

    # Khoi tao SessionTracker (Phần 2) de quan ly toan bo CSV & Do thi
    tracker = SessionTracker(logs_dir=LOGS_DIR)

    print("\n" + "=" * 65)
    print("[HE THONG] DANG KET NOI VOI TRO LY QUY CHE & THU TUC MCP SERVER...")
    print("=" * 65)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Dynamic tool discovery (MCP Specification Week 7)
            mcp_tools = (await session.list_tools()).tools

            # Tool denylist: Cac tool quan tri / tai lieu toan van danh rieng cho lenh CLI (/docs, /sync)
            # Khong day vao groq_tools de giam token schema cho LLM (giai quyet triet de loi 429/413)
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

            print("[THANH CONG] Da ket noi MCP Server!")
            print(f"Cac cong cu agent duoc phep su dung ({len(groq_tools)}): {', '.join([t['function']['name'] for t in groq_tools])}")
            print("\nCac lenh tien ich nhanh:")
            print("   - /docs  : Xem danh muc cac van ban quy che hien hanh")
            print("   - /sync  : Quet va dong bo lai thu muc quy che data/")
            print("   - /eval  : Tham dinh do trung thuc (Faithfulness) cua cau vua tra loi")
            print("   - /chart : Xuat file CSV va ve bieu do phien lam viec ngay lap tuc")
            print("   - /clear : Lam moi phien lam viec")
            print("   - exit   : Thoat chuong trinh va tu dong luu bao cao phien")
            print("=" * 65)

            messages = [{"role": "system", "content": system_prompt}]
            evaluator = FaithfulnessEvaluator(groq_client=groq_client)
            last_answer = ""
            last_context = ""
            last_question = ""

            try:
                while True:
                    try:
                        user_input = input("\nSinh vien / Nguoi dung: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nTam biet ban!")
                        break

                    if not user_input:
                        continue

                    exit_cmds = {"exit", "quit", ":q", "q", "thoat", "thoát", "ẽit", "exi", "eit", "exxit", "exti", "out", "/exit", "/quit"}
                    if user_input.lower() in exit_cmds:
                        print("\nDa ket thuc phien lam viec. Chuc ban hoc tap tot!")
                        break

                    if user_input.lower() in ["/clear", "clear"]:
                        messages = [{"role": "system", "content": system_prompt}]
                        last_answer = ""
                        last_context = ""
                        last_question = ""
                        print("[THANH CONG] Da lam moi phien hoi thoai.")
                        continue

                    if user_input.lower() in ["/docs", "/list"]:
                        print("\n[DANG XU LY] Dang tra cuu danh muc van ban quy che...")
                        res = await session.call_tool("list_indexed_documents", arguments={})
                        print(res.content[0].text)
                        continue

                    if user_input.lower() in ["/sync", "/reload", "sync"]:
                        print("\n[DANG XU LY] Dang dong bo lai thu muc data/...")
                        res = await session.call_tool("sync_documents", arguments={})
                        print(res.content[0].text)
                        continue

                    # Xuất báo cáo CSV và đồ thị tức thì qua SessionTracker
                    if user_input.lower() in ["/chart", "/charts", "chart"]:
                        if tracker.is_empty():
                            print("[Canh bao] Chua co cau hoi nao trong phien de ve bieu do.")
                            continue
                        print("\n[DANG XU LY] Dang xuat CSV va ve do thi cho phien lam viec...")
                        report_out = tracker.export_report(is_manual=True)
                        if report_out:
                            print(f"  -> File CSV phien: {report_out['csv_path']}")
                            for chart_path in report_out["charts"]:
                                print(f"  -> Bieu do phien : {chart_path}")
                        continue

                    # Thẩm định chất lượng câu trả lời bằng LLM-as-a-judge
                    if user_input.lower() in ["/eval", "eval"]:
                        if not last_answer:
                            print("[Canh bao] Chua co cau tra loi nao de tham dinh. Hay hoi mot cau truoc.")
                            continue
                        print("\n[DANG XU LY] Dang danh gia do trung thuc (Faithfulness / LLM-as-a-judge)...")
                        report = evaluator.evaluate(last_answer, last_context)
                        print(report.format_terminal())

                        if last_question:
                            try:
                                rel_score = check_answer_relevance(last_question, last_answer, groq_client=groq_client)
                                rel_pct = int(round(rel_score * 100))
                                rel_badge = "[ĐẠT TRỌNG TÂM]" if rel_score >= 0.7 else "[CẦN CẢI THIỆN]"
                                print(f"-> Độ liên quan câu hỏi (Answer Relevance): {rel_pct}% | {rel_badge}")
                            except Exception:
                                pass

                        # Cập nhật điểm chất lượng vào SessionTracker
                        tracker.update_last_faithfulness(report.score)
                        continue

                    # Bắt đầu lượt hỏi đáp với Agent Loop
                    messages.append({"role": "user", "content": user_input})
                    print("\n[DANG XU LY] Tro ly AI dang phan tich quy che va suy luan...")

                    start_time = time.time()
                    answer, context, usage = await run_agent_loop(
                        session=session,
                        groq_client=groq_client,
                        groq_tools=groq_tools,
                        messages=messages,
                        user_query=user_input,
                        tracker=tracker,
                    )
                    duration = time.time() - start_time
                    last_answer = answer
                    last_context = context
                    last_question = user_input

                    # Ghi nhận lượt hỏi đáp vào SessionTracker (Phần 2)
                    tracker.record_turn(
                        query=user_input,
                        answer=answer,
                        duration=duration,
                        usage=usage,
                    )

                    print("\n" + "-" * 65)
                    print(f"KET QUA TRA LOI:\n{answer}")
                    print("-" * 65)
                    print(f"[Chi phi phien]: {usage['total_tokens']} tokens ({usage['api_calls']} luot goi API)")
                    print("Goi y: Go /eval de tham dinh trung thuc, go /chart de xuat bieu do phien.")

            finally:
                # Tự động xuất CSV và vẽ đồ thị khi kết thúc phiên
                if not tracker.is_empty():
                    print("\n" + "=" * 65)
                    print("[BAO CAO PHIEN] Dang tu dong luu log CSV va ve bieu do phien...")
                    report_out = tracker.export_report()
                    if report_out:
                        print(f"- Log CSV phien  : {report_out['csv_path']}")
                        for chart_path in report_out["charts"]:
                            print(f"- Bieu do phien : {chart_path}")
                    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
