"""
Terminal RAG MCP Client with ReAct Agent Loop (Week 6 & Week 7).
Connects to rag_server.py over stdio and interacts with user in the terminal.
Run using: uv run python rag_client.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
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
MAX_ITERATIONS = 5

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "rag_server.py")],
)

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp phân tích tài liệu nội bộ (RAG Assistant).
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA HOÀN TOÀN VÀO CÁC TÀI LIỆU TRONG HỆ THỐNG.

QUY TẮC BẮT BUỘC (STRICT GROUNDING & ANTI-HALLUCINATION):
1. BẮT BUỘC DÙNG TOOL: Với mọi câu hỏi của người dùng, bạn PHẢI LUÔN LUÔN gọi công cụ `search_documents` để tìm thông tin trước. TUYỆT ĐỐI KHÔNG ĐƯỢC trả lời từ kiến thức cá nhân có sẵn.
2. NGUỒN GỐC THẬT 100%: Bạn CHỈ ĐƯỢC PHÉP trích dẫn tên file và Chunk ID thực sự xuất hiện trong kết quả của tool `search_documents`. TUYỆT ĐỐI CẤM TỰ BỊA ĐẶT TÊN SÁCH, TÊN FILE HOẶC CHUNK ID (như "Understanding English Writing" hay "eng_ch...").
3. TRÍCH DẪN CUỐI CÂU: Mỗi câu trả lời bắt buộc phải có trích dẫn thật:
   [Nguồn: <Tên file trong kết quả> | Chunk ID: <Chunk ID thực tế>]
4. TỪ CHỐI NẾU KHÔNG CÓ: Nếu kết quả tìm kiếm không chứa thông tin cần tìm, hãy trung thực trả lời: "Tôi không tìm thấy thông tin này trong các tài liệu hiện có của bạn." Không được tự suy diễn.
"""


async def run_agent_turn(
    session: ClientSession,
    groq_client: Groq,
    groq_tools: list[dict],
    messages: list[dict],
) -> tuple[str, str]:
    """Execute ReAct Agent Loop. Returns (final_answer, accumulated_context)."""
    accumulated_context = []

    for iteration in range(MAX_ITERATIONS):
        # Ở lượt đầu tiên của câu hỏi mới, BẮT BUỘC Model phải gọi tool tra cứu (Strict Grounding)
        current_tool_choice = "required" if iteration == 0 else "auto"

        # 1. Gọi LLM
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=groq_tools,
            tool_choice=current_tool_choice,
            max_tokens=1500,
        )

        response_message = response.choices[0].message

        # 2. Kiểm tra xem LLM có yêu cầu gọi Tool không
        if response_message.tool_calls:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_args = {}

                print(f"\n⚙️  [Agent Action] Gọi công cụ: '{tool_name}'...")
                if tool_name == "search_documents":
                    print(f"    ↳ Query: \"{tool_args.get('query')}\"")

                # 3. Gọi Tool trên MCP Server qua stdio
                try:
                    tool_result = await session.call_tool(tool_name, arguments=tool_args)
                    result_text = tool_result.content[0].text if tool_result.content else ""
                except Exception as e:
                    result_text = f"Lỗi khi thực thi tool: {e}"

                print(f"    ↳ Nhận kết quả: {len(result_text)} ký tự.")
                accumulated_context.append(result_text)

                # 4. Gửi kết quả quan sát (Observation) lại cho LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

            continue

        # 5. Nếu không gọi tool -> Đã có câu trả lời cuối cùng
        final_answer = response_message.content or ""
        messages.append({"role": "assistant", "content": final_answer})
        return final_answer, "\n\n".join(accumulated_context)

    return "⚠️ Quá số vòng lặp cho phép (Iteration cap reached). Vui lòng thử lại với câu hỏi cụ thể hơn.", "\n\n".join(accumulated_context)


async def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Lỗi: Chưa tìm thấy GROQ_API_KEY trong file .env")
        return

    groq_client = Groq(api_key=api_key)

    print("\n" + "=" * 65)
    print("🤖 ĐANG KẾT NỐI VỚI RAG MCP SERVER...")
    print("=" * 65)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Khám phá Tool động từ MCP Server (Week 7)
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

            print("✅ Đã kết nối MCP Server thành công!")
            print(f"🔧 Các công cụ sẵn sàng: {', '.join([t.name for t in mcp_tools])}")
            print("\n💡 Các lệnh tiện ích:")
            print("   • /docs : Xem danh sách tài liệu hiện có")
            print("   • /sync : Quét và nạp ngay các file mới được thêm vào data/")
            print("   • /eval : Thẩm định độ trung thực của câu trả lời vừa rồi (Week 5)")
            print("   • /clear: Xóa lịch sử ngữ cảnh cuộc trò chuyện")
            print("   • exit  : Thoát chương trình")
            print("=" * 65)

            # Lịch sử hội thoại & bộ thẩm định
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            evaluator = FaithfulnessEvaluator(groq_client=groq_client)
            last_answer = ""
            last_context = ""

            while True:
                try:
                    user_input = input("\n👤 Bạn: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nTạm biệt!")
                    break

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", ":q"]:
                    print("\nĐã ngắt kết nối. Hẹn gặp lại bạn!")
                    break

                if user_input.lower() in ["/clear", "clear"]:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    last_answer = ""
                    last_context = ""
                    print("🧹 Đã làm mới lịch sử cuộc trò chuyện.")
                    continue

                if user_input.lower() in ["/docs", "/list"]:
                    print("\n📚 Đang tra cứu danh mục tài liệu...")
                    res = await session.call_tool("list_indexed_documents", arguments={})
                    print(res.content[0].text)
                    continue

                if user_input.lower() in ["/sync", "/reload", "sync"]:
                    print("\n🔄 Đang đồng bộ hóa thư mục data/...")
                    res = await session.call_tool("sync_documents", arguments={})
                    print(res.content[0].text)
                    continue

                if user_input.lower() in ["/eval", "eval"]:
                    if not last_answer:
                        print("⚠️ Chưa có câu trả lời nào để thẩm định. Hãy đặt một câu hỏi trước.")
                        continue
                    print("\n🛡️ Đang phân tích mệnh đề và thẩm định độ trung thực (LLM-as-a-judge)...")
                    report = evaluator.evaluate(last_answer, last_context)
                    print(report.format_terminal())
                    continue

                # Thêm tin nhắn người dùng vào messages
                messages.append({"role": "user", "content": user_input})

                print("\n🤖 Trợ lý AI đang suy luận...")
                answer, context = await run_agent_turn(
                    session=session,
                    groq_client=groq_client,
                    groq_tools=groq_tools,
                    messages=messages,
                )
                last_answer = answer
                last_context = context

                print("\n" + "─" * 60)
                print(f"📝 TRẢ LỜI:\n{answer}")
                print("─" * 60)
                print("💡 Mẹo: Gõ /eval để kiểm tra xem câu trả lời trên có chuẩn xác theo tài liệu không.")


if __name__ == "__main__":
    asyncio.run(main())
