"""Kiểm tra phản hồi của Agent khi nhận các câu hỏi về năm (năm cụ thể, hiện tại, so sánh các năm).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Đảm bảo UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from rag_client import run_agent_loop, load_system_prompt, SERVER_PARAMS

TEST_PROMPTS = [
    {
        "id": "PROMPT_1_PAST_YEAR_2023",
        "description": "Hỏi quy định năm cũ (2023) về hiến máu tình nguyện",
        "query": "Theo quy định năm 2023, trực tiếp hiến máu tình nguyện được quy đổi bao nhiêu ngày CTXH và điểm số là bao nhiêu?",
        "expected_facts": ["0,5", "5"],
        "expected_year": "2023",
    },
    {
        "id": "PROMPT_2_NEW_YEAR_2025",
        "description": "Hỏi quy định năm mới (2025) về hiến máu tình nguyện",
        "query": "Theo quy định năm 2025, trực tiếp hiến máu được quy đổi bao nhiêu ngày CTXH và thang điểm là bao nhiêu?",
        "expected_facts": ["1,0", "12"],
        "expected_year": "2025",
    },
    {
        "id": "PROMPT_3_DEFAULT_LATEST",
        "description": "Hỏi quy định hiện tại (không nêu năm, hệ thống phải tự lấy bản mới nhất 2025)",
        "query": "Mức thưởng tiền mặt của danh hiệu Chiến sĩ thi đua cấp Trường hiện nay là bao nhiêu?",
        "expected_facts": ["1.500.000"],
        "expected_year": "2025",
    },
    {
        "id": "PROMPT_4_COMPARE_YEARS",
        "description": "So sánh mức thưởng Chiến sĩ thi đua cấp Trường giữa năm 2023 và năm 2025",
        "query": "So sánh mức thưởng tiền mặt danh hiệu Chiến sĩ thi đua cấp Trường giữa năm 2023 và năm 2025?",
        "expected_facts": ["500.000", "1.500.000"],
        "expected_year": "both",
    },
]

async def run_single_test(session, groq_client, groq_tools, sys_prompt, item):
    print("=" * 80)
    print(f"TEST: {item['id']} - {item['description']}")
    print(f"Câu hỏi: \"{item['query']}\"")
    print("-" * 80)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": item["query"]},
    ]

    answer, context, tokens = await run_agent_loop(
        groq_client=groq_client,
        session=session,
        messages=messages,
        groq_tools=groq_tools,
        user_query=item["query"],
    )

    print("\n--- PHẢN HỒI CỦA AGENT ---")
    print(answer)
    print("\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC ---")
    
    # Kiểm tra các dữ kiện mong đợi
    passed_facts = []
    missing_facts = []
    for fact in item["expected_facts"]:
        if fact.lower() in answer.lower():
            passed_facts.append(fact)
        else:
            missing_facts.append(fact)

    facts_ok = len(missing_facts) == 0
    print(f"- Dữ kiện mong đợi: {item['expected_facts']}")
    print(f"- Dữ kiện tìm thấy trong câu trả lời: {passed_facts}")
    if missing_facts:
        print(f"- Dữ kiện thiếu: {missing_facts}")

    print(f"- Kết quả tổng kết: {'✅ ĐẠT (PASS)' if facts_ok else '❌ CHƯA ĐẠT (FAIL)'}")
    print("=" * 80 + "\n")
    return facts_ok, answer

async def main():
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
            groq_client = Groq()
            sys_prompt = load_system_prompt()

            results = []
            for item in TEST_PROMPTS:
                ok, ans = await run_single_test(session, groq_client, groq_tools, sys_prompt, item)
                results.append((item["id"], ok))

            print("\n" + "#" * 40)
            print("TỔNG KẾT TOÀN BỘ CÁC BÀI TEST VỀ NĂM:")
            print("#" * 40)
            all_passed = True
            for tid, ok in results:
                status = "✅ PASS" if ok else "❌ FAIL"
                print(f"  {tid:<30}: {status}")
                if not ok:
                    all_passed = False
            print(f"\nKẾT LUẬN CHUNG: {'TẤT CẢ ĐỀU ĐẠT' if all_passed else 'CÓ BÀI TEST CHƯA ĐẠT'}")

if __name__ == "__main__":
    asyncio.run(main())
