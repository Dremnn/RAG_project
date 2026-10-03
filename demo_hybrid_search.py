"""
Demo script to test unified VectorStore with built-in Hybrid Search.
Run using: uv run python demo_hybrid_search.py
"""

from pathlib import Path
import sys

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.loaders import DocumentLoaderFactory
from src.chunking import RecursiveChunker
from src.retrieval import EmbeddingGenerator, VectorStore


def main():
    data_dir = CURRENT_DIR / "data"
    index_file = data_dir / "index.json"

    print("=" * 70)
    print("🔍 TESTING UNIFIED VECTOR STORE WITH BUILT-IN HYBRID SEARCH")
    print("=" * 70)

    # 1. Initialize or load VectorStore
    if index_file.exists():
        print(f"📦 Loading cached vector store from: {index_file.name}")
        store = VectorStore.load_from_json(index_file)
        print(f"✅ Loaded {len(store.chunks)} indexed chunks.")
    else:
        print("📄 Step 1: Loading documents from data/ ...")
        docs = DocumentLoaderFactory.load_directory(data_dir)
        print(f"   Loaded {len(docs)} documents.")

        print("✂️ Step 2: Chunking documents (size=600, overlap=80) ...")
        chunker = RecursiveChunker(chunk_size=600, chunk_overlap=80)
        chunks = chunker.split_documents(docs)
        print(f"   Created {len(chunks)} chunks.")

        print("🧠 Step 3: Generating embeddings via Jina AI API ...")
        emb_gen = EmbeddingGenerator()
        texts_to_embed = [c.content for c in chunks]
        embeddings = emb_gen.embed_documents(texts_to_embed, batch_size=16)

        store = VectorStore()
        store.add_chunks(chunks, embeddings)
        store.save_to_json(index_file)
        print(f"💾 Saved {len(store.chunks)} chunks to {index_file.name}")

    # 2. Test Queries using store.search() directly!
    test_queries = [
        "Quy định làm việc từ xa và phụ cấp thiết bị cho nhân viên",
        "What are the aims and key findings in writing an abstract?",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"❓ QUERY: \"{q}\"")
        print("-" * 70)

        # Call store.search with mode='hybrid' directly!
        results = store.search(q, top_k=3, mode="hybrid")

        for rank, (chunk, score) in enumerate(results, 1):
            source_file = chunk.metadata.get("file_name", "unknown")
            page_info = f" (Page {chunk.metadata.get('page_number')})" if "page_number" in chunk.metadata else ""

            print(f"Top {rank} | Score: {score:.4f} | Chunk ID: {chunk.chunk_id} | Source: {source_file}{page_info}")
            print("  • Content snippet:")
            snippet = chunk.content.replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            print(f"    \"{snippet}\"\n")


if __name__ == "__main__":
    main()
