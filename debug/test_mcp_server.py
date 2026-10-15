"""
Verification script for RAG MCP Server over stdio.
Run using: uv run python test_mcp_server.py
"""

import asyncio
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "debug" else CURRENT_DIR

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_ROOT / "rag_server.py")],
)


async def main():
    print("=" * 65)
    print("CONNECTING TO RAG MCP SERVER OVER STDIO...")
    print("=" * 65)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[OK] MCP Session Initialized successfully!\n")

            # 1. Discover tools via list_tools() (W7 Dynamic Discovery)
            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"Discovered {len(tools)} MCP Tools:")
            for t in tools:
                print(f"  - Tool: {t.name}")
                print(f"    Description: {t.description[:80]}...")
            print("-" * 65)

            # 2. Call tool: check_student_exists
            print("\nCalling Tool: 'check_student_exists' (mssv='22110001'):")
            stu_check = await session.call_tool("check_student_exists", arguments={"mssv": "22110001"})
            print(stu_check.content[0].text)
            print("-" * 65)

            # 3. Call tool: check_graduation_eligibility
            print("\nCalling Tool: 'check_graduation_eligibility' (mssv='22110001'):")
            grad_check = await session.call_tool("check_graduation_eligibility", arguments={"mssv": "22110001"})
            print(grad_check.content[0].text)
            print("-" * 65)

            # 4. Call tool: search_ctxh_rules
            query = "Trực tiếp hiến máu tình nguyện"
            print(f"\nCalling Tool: 'search_ctxh_rules' (query: '{query}'):")
            search_res = await session.call_tool(
                "search_ctxh_rules",
                arguments={"query": query, "top_k": 2},
            )
            print(search_res.content[0].text[:300] + "...")
            print("=" * 65)
            print("[OK] MCP SERVER VERIFICATION PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
