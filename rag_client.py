"""
Terminal RAG MCP Client with ReAct Agent Loop (Week 6 & Week 7).
Connects to rag_server.py over stdio and interacts with user in the terminal.
Designed for: Tro ly Quy che & Thu tuc Sinh vien.
Run using: uv run python rag_client.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.evaluation import FaithfulnessEvaluator

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

HERE = Path(__file__).resolve().parent
MODEL = "qwen/qwen3.8-27b"
MAX_STEPS = 6
LOGS_DIR = HERE / "logs"
PROMPT_PATH = HERE / "prompts" / "system_agent.txt"

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "rag_server.py")],
)


def load_system_prompt() -> str:
    """Load system prompt from external file (Week 2 clean architecture)."""
    if PROMPT_PATH.exists():
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return (
        "Ban la Tro ly AI Quy che & Thu tuc Sinh vien. "
        "Luon tra cuu van ban quy che qua search_regulations va trich dan nguon [Nguon: ... | Chunk ID: ...]."
    )


def log_agent_run(run_record: dict) -> None:
    """Save execution trace and token usage to logs/*.jsonl (Week 5 & Week 7 requirement)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    log_file = LOGS_DIR / f"run_{today_str}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record, ensure_ascii=False) + "\n")


def call_groq_with_retry(
    groq_client: Groq,
    messages: list[dict],
    tools: list[dict],
    tool_choice: str = "auto",
    max_retries: int = 3,
    timeout: float = 30.0,
) -> tuple[any, dict]:
    """Call Groq API with exponential backoff retry and timeout handling (Week 2 & Week 6)."""
    last_err = None
    delay = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=1500,
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
            print(f"[Canh bao] Groq API gap su co (Thu lan {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(f"Goi Groq API that bai sau {max_retries} lan thu: {last_err}")


async def run_agent_loop(
    session: ClientSession,
    groq_client: Groq,
    groq_tools: list[dict],
    messages: list[dict],
    user_query: str,
) -> tuple[str, str, dict]:
    """
    Custom ReAct Agent Loop (Week 6 & Week 7).
    - MAX_STEPS = 6
    - Strict Grounding: First step enforces tool call
    - Logs thoughts, actions, observations, and token usage.
    """
    accumulated_context = []
    trajectory = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

    for step in range(MAX_STEPS):
        # Strict Grounding: Bat buoc goi tool o buoc dau tien de tra cuu hoac xu ly
        tool_choice = "required" if step == 0 else "auto"

        try:
            response, usage = call_groq_with_retry(
                groq_client=groq_client,
                messages=messages,
                tools=groq_tools,
                tool_choice=tool_choice,
            )
        except Exception as e:
            err_msg = f"Loi ket noi LLM: {e}"
            print(f"[Loi] {err_msg}")
            return err_msg, "\n\n".join(accumulated_context), total_tokens

        total_tokens["prompt_tokens"] += usage["prompt_tokens"]
        total_tokens["completion_tokens"] += usage["completion_tokens"]
        total_tokens["total_tokens"] += usage["total_tokens"]
        total_tokens["api_calls"] += 1

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
                if tool_name in ["search_regulations", "search_documents"]:
                    print(f"    -> Tu khoa: \"{tool_args.get('query')}\"")
                elif tool_name == "submit_academic_request":
                    print(f"    -> [GHI DU LIEU] MSSV: {tool_args.get('mssv')} | Loai: {tool_args.get('request_type')}")
                elif tool_name == "get_request_status":
                    print(f"    -> [KIEM CHUNG] Ma don can verify: {tool_args.get('request_id')}")

                # Goi Tool tren MCP Server qua stdio an toan (khong crash)
                try:
                    tool_result = await session.call_tool(tool_name, arguments=tool_args)
                    result_text = tool_result.content[0].text if tool_result.content else ""
                except Exception as e:
                    result_text = f"Loi khi thuc thi cong cu '{tool_name}': {e}"

                print(f"    -> Nhan phan hoi: {len(result_text)} ky tu.")
                accumulated_context.append(result_text)

                trajectory.append({
                    "step": step + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": result_text[:200] if len(result_text) > 200 else result_text,
                })

                # Gui ket qua quan sat (Observation) lai cho LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

            continue

        # Neu khong con tool_calls -> LLM da hoan tat suy luan va tra ve cau tra loi cuoi cung
        final_answer = response_message.content or ""
        messages.append({"role": "assistant", "content": final_answer})

        # Ghi log jsonl
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "steps": len(trajectory),
            "trajectory": trajectory,
            "final_answer": final_answer,
            "usage": total_tokens,
        }
        log_agent_run(log_record)

        return final_answer, "\n\n".join(accumulated_context), total_tokens

    # Qua so buoc cho phep (Max steps reached)
    warning_msg = (
        "[Thong bao] Da vuot qua gioi han 6 buoc xu ly cho phep. "
        "Vui long cung cap yeu cau cu the hon de tro ly phuc vu tot hon."
    )
    log_record = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "steps": MAX_STEPS,
        "trajectory": trajectory,
        "final_answer": warning_msg,
        "usage": total_tokens,
        "status": "MAX_STEPS_REACHED",
    }
    log_agent_run(log_record)

    return warning_msg, "\n\n".join(accumulated_context), total_tokens


async def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Loi] Chua tim thay GROQ_API_KEY trong file .env")
        return

    groq_client = Groq(api_key=api_key)
    system_prompt = load_system_prompt()

    print("\n" + "=" * 65)
    print("[HE THONG] DANG KET NOI VOI TRO LY QUY CHE & THU TUC MCP SERVER...")
    print("=" * 65)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Dynamic tool discovery (MCP Specification Week 7)
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

            print("[THANH CONG] Da ket noi MCP Server!")
            print(f"Cac cong cu duoc phep goi: {', '.join([t.name for t in mcp_tools])}")
            print("\nCac lenh tien ich nhanh:")
            print("   - /docs : Xem danh muc cac van ban quy che hien hanh")
            print("   - /sync : Quet va dong bo lai thu muc quy che data/")
            print("   - /eval : Tham dinh do trung thuc (Faithfulness) cua cau vua tra loi")
            print("   - /clear: Lam moi phien lam viec")
            print("   - exit  : Thoat chuong trinh")
            print("=" * 65)

            messages = [{"role": "system", "content": system_prompt}]
            evaluator = FaithfulnessEvaluator(groq_client=groq_client)
            last_answer = ""
            last_context = ""

            while True:
                try:
                    user_input = input("\nSinh vien / Nguoi dung: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nTam biet ban!")
                    break

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", ":q"]:
                    print("\nDa ket thuc phien lam viec. Chuc ban hoc tap tot!")
                    break

                if user_input.lower() in ["/clear", "clear"]:
                    messages = [{"role": "system", "content": system_prompt}]
                    last_answer = ""
                    last_context = ""
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

                if user_input.lower() in ["/eval", "eval"]:
                    if not last_answer:
                        print("[Canh bao] Chua co cau tra loi nao de tham dinh. Hay hoi mot cau truoc.")
                        continue
                    print("\n[DANG XU LY] Dang danh gia do trung thuc (Faithfulness / LLM-as-a-judge)...")
                    report = evaluator.evaluate(last_answer, last_context)
                    print(report.format_terminal())
                    continue

                messages.append({"role": "user", "content": user_input})
                print("\n[DANG XU LY] Tro ly AI dang phan tich quy che va suy luan...")

                answer, context, usage = await run_agent_loop(
                    session=session,
                    groq_client=groq_client,
                    groq_tools=groq_tools,
                    messages=messages,
                    user_query=user_input,
                )
                last_answer = answer
                last_context = context

                print("\n" + "-" * 65)
                print(f"KET QUA TRA LOI:\n{answer}")
                print("-" * 65)
                print(f"[Chi phi phien]: {usage['total_tokens']} tokens ({usage['api_calls']} luot goi API)")
                print("Goi y: Go /eval de tham dinh xem cau tra loi tren co trung thuc voi quy che khong.")


if __name__ == "__main__":
    asyncio.run(main())
