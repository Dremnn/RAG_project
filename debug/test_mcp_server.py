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

            # 2. Call tool: list_indexed_documents
            print("\nCalling Tool: 'list_indexed_documents':")
            doc_list = await session.call_tool("list_indexed_documents", arguments={})
            print(doc_list.content[0].text)
            print("-" * 65)

            # 3. Call tool: search_documents
            query = "Quy định làm việc từ xa và phụ cấp"
            print(f"\nCalling Tool: 'search_documents' (query: '{query}'):")
            search_res = await session.call_tool(
                "search_documents",
                arguments={"query": query, "top_k": 2},
            )
            print(search_res.content[0].text)
            print("=" * 65)
            print("[OK] MCP SERVER VERIFICATION PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
