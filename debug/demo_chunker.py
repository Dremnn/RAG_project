"""
Demo script to test Chunking strategies on loaded PDF, DOCX, and TXT documents.
Run using: uv run python demo_chunker.py
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
from src.chunking import RecursiveChunker, FixedSizeChunker


def main():
    data_dir = CURRENT_DIR / "data"
    print("=" * 65)
    print("TESTING CHUNKING STRATEGY (Recursive Character Splitting)")
    print("=" * 65)

    # 1. Load documents using our factory from Part 1
    docs = DocumentLoaderFactory.load_directory(data_dir)
    print(f"Loaded {len(docs)} documents from data/\n")

    # 2. Initialize RecursiveChunker
    # For demonstration, chunk_size=250 chars, overlap=40 chars
    chunker = RecursiveChunker(chunk_size=250, chunk_overlap=40)
    all_chunks = chunker.split_documents(docs)

    print(f"Total chunks created: {len(all_chunks)}")
    print("-" * 65)

    # Group and display chunks per document
    doc_chunk_map = {}
    for chunk in all_chunks:
        doc_chunk_map.setdefault(chunk.doc_id, []).append(chunk)

    for doc_id, chunks in doc_chunk_map.items():
        doc_title = chunks[0].metadata.get("title", doc_id)
        doc_ext = chunks[0].metadata.get("extension", "")
        print(f"\nDocument: '{doc_title}' ({doc_ext}) -> {len(chunks)} chunks:")
        for c in chunks:
            preview = c.content.replace("\n", " ")
            if len(preview) > 90:
                preview = preview[:90] + "..."
            print(f"   [{c.chunk_id}] (len: {len(c.content)} chars, words: {c.metadata.get('word_count')}):")
            print(f"      \"{preview}\"")

    # 3. Demonstrate Overlap between two consecutive chunks
    if len(all_chunks) >= 2:
        print("\n" + "=" * 65)
        print("DEMONSTRATING OVERLAP (Goi dau giua Chunk 0 va Chunk 1):")
        c0 = all_chunks[0]
        c1 = all_chunks[1]
        print(f"Chunk 0 End:   ...{c0.content[-40:].strip()}")
        print(f"Chunk 1 Start: {c1.content[:40].strip()}...")
        print("=" * 65)


if __name__ == "__main__":
    main()
