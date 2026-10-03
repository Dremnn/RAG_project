import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

# Set UTF-8 encoding for standard streams
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.loaders import DocumentLoaderFactory, SUPPORTED_EXTENSIONS
from src.chunking import RecursiveChunker
from src.retrieval import VectorStore, EmbeddingGenerator

load_dotenv()

# Initialize MCP Server
mcp = MCPServer("rag-knowledge-server")

DATA_DIR = _HERE / "data"
INDEX_PATH = DATA_DIR / "index.json"

# Global Store
STORE = VectorStore()


def sync_knowledge_base() -> dict:
    """
    Incremental synchronization of data/ directory:
    - Finds new files -> chunks, embeds, adds to store.
    - Finds modified files -> removes old chunks, re-chunks, re-embeds.
    - Finds deleted files -> removes chunks from store.
    - Saves updated store to index.json.
    """
    global STORE
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover all supported files currently on disk in data/
    current_files = {}
    for p in DATA_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            current_files[p.name] = (p, p.stat().st_mtime)

    # Populate file_meta if missing from existing chunks
    if not STORE.file_meta and STORE.chunks:
        for chunk in STORE.chunks.values():
            fname = chunk.metadata.get("file_name")
            if fname and fname in current_files and fname not in STORE.file_meta:
                STORE.file_meta[fname] = current_files[fname][1]

    added_files = []
    updated_files = []
    removed_files = []

    # 2. Check for deleted files (was in store, but no longer on disk)
    for indexed_file in list(STORE.file_meta.keys()):
        if indexed_file not in current_files:
            STORE.remove_chunks_by_file(indexed_file)
            removed_files.append(indexed_file)

    # 3. Check for new or modified files
    emb_gen = None
    chunker = RecursiveChunker(chunk_size=600, chunk_overlap=80)

    for fname, (fpath, mtime) in current_files.items():
        is_new = fname not in STORE.file_meta
        is_modified = not is_new and mtime > STORE.file_meta[fname]

        if is_new or is_modified:
            if is_modified:
                STORE.remove_chunks_by_file(fname)
                updated_files.append(fname)
            else:
                added_files.append(fname)

            # Load and chunk this file only
            docs = DocumentLoaderFactory.load_file(fpath)
            chunks = chunker.split_documents(docs)

            if chunks:
                if emb_gen is None:
                    emb_gen = EmbeddingGenerator()
                texts = [c.content for c in chunks]
                embeddings = emb_gen.embed_documents(texts, batch_size=16)
                STORE.add_chunks(chunks, embeddings)

            STORE.file_meta[fname] = mtime

    # 4. Save to index.json if any changes occurred
    if added_files or updated_files or removed_files or not INDEX_PATH.exists():
        STORE.save_to_json(INDEX_PATH)

    return {
        "added": added_files,
        "updated": updated_files,
        "removed": removed_files,
        "total_chunks": len(STORE.chunks),
        "total_files": len(STORE.file_meta),
    }


def _ensure_index_loaded() -> None:
    """Load cached index and incrementally synchronize with data/."""
    global STORE
    if INDEX_PATH.exists():
        try:
            STORE = VectorStore.load_from_json(INDEX_PATH)
        except Exception as e:
            STORE = VectorStore()
    
    # Sync any new or modified files in data/
    sync_knowledge_base()


# Initialize and sync store on startup
_ensure_index_loaded()


# ==========================================
# MCP TOOLS (Declared for AI Agents to call)
# ==========================================

@mcp.tool()
def search_documents(query: str, top_k: int = 4) -> str:
    """Search the document corpus using Hybrid Search (BM25 keyword matching + Dense Vector semantics).
    Returns the most relevant text chunks with citations, source files, and relevance scores.
    Always call this tool when answering factual questions about user documents."""
    top_k = max(1, min(top_k, 8))
    results = STORE.search(query=query, top_k=top_k, mode="hybrid")

    if not results:
        return "No relevant passages found in the document corpus."

    formatted_passages = []
    for rank, (chunk, score) in enumerate(results, 1):
        meta = chunk.metadata or {}
        file_name = meta.get("file_name", "unknown")
        page_str = f" | Page {meta.get('page_number')}" if "page_number" in meta else ""
        
        passage = (
            f"--- [Passage {rank}] ---\n"
            f"Source: {file_name}{page_str}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Relevance Score: {score:.4f}\n"
            f"Content:\n{chunk.content}"
        )
        formatted_passages.append(passage)

    return "\n\n".join(formatted_passages)


@mcp.tool()
def list_indexed_documents() -> str:
    """List all documents currently indexed in the knowledge base,
    including file names, extensions, and total chunks.
    Use this to orient yourself on what topics are available."""
    if not STORE.chunks:
        return "The document knowledge base is empty."

    # Group by source document
    doc_summary: dict[str, dict] = {}
    for chunk in STORE.chunks.values():
        meta = chunk.metadata or {}
        doc_id = chunk.doc_id
        if doc_id not in doc_summary:
            doc_summary[doc_id] = {
                "file_name": meta.get("file_name", doc_id),
                "extension": meta.get("extension", ""),
                "chunk_count": 0,
                "total_pages": meta.get("total_pages"),
            }
        doc_summary[doc_id]["chunk_count"] += 1

    lines = ["Available Documents in Corpus:"]
    for doc_id, info in doc_summary.items():
        pages_info = f", Pages: {info['total_pages']}" if info["total_pages"] else ""
        lines.append(
            f"• [{doc_id}] {info['file_name']} (Format: {info['extension']}, Chunks: {info['chunk_count']}{pages_info})"
        )

    return "\n".join(lines)


@mcp.tool()
def get_document_full(doc_id: str) -> str:
    """Retrieve all chunks belonging to a specific document ID in reading order.
    Use after search_documents when you need full document context for deep comprehension."""
    matched_chunks = sorted(
        [c for c in STORE.chunks.values() if c.doc_id == doc_id],
        key=lambda c: c.chunk_index,
    )

    if not matched_chunks:
        return f"No document found with ID '{doc_id}'."

    meta = matched_chunks[0].metadata or {}
    file_name = meta.get("file_name", doc_id)

    full_text = "\n\n".join(c.content for c in matched_chunks)
    if len(full_text) > 4000:
        full_text = full_text[:4000] + "\n[ ... truncated for context safety ... ]"

    return f"=== Full Document: {file_name} ({doc_id}) ===\n\n{full_text}"


@mcp.tool()
def sync_documents() -> str:
    """Rescan the data/ directory for new, updated, or deleted documents
    and incrementally update the knowledge index without re-indexing unchanged files.
    Call this when the user asks to refresh, reload, or sync newly added files."""
    report = sync_knowledge_base()
    lines = ["Knowledge Base Synchronization Report:"]
    if report["added"]:
        lines.append(f"  • Added new files: {', '.join(report['added'])}")
    if report["updated"]:
        lines.append(f"  • Updated modified files: {', '.join(report['updated'])}")
    if report["removed"]:
        lines.append(f"  • Removed deleted files: {', '.join(report['removed'])}")
    if not (report["added"] or report["updated"] or report["removed"]):
        lines.append("  • All documents are already up-to-date. No changes detected.")
    lines.append(f"  • Current Status: {report['total_files']} files, {report['total_chunks']} total chunks.")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")

