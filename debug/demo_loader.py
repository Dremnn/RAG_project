"""
Demo script to test loading PDF, DOCX, TXT documents using DocumentLoaderFactory.
Run using: uv run python demo_loader.py
"""

from pathlib import Path
import sys

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure RAG_project directory is in PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.loaders import DocumentLoaderFactory, PDFLoader, DocxLoader, TextLoader


def main():
    data_dir = CURRENT_DIR / "data"
    print("=" * 60)
    print("TESTING RAG DOCUMENT LOADERS (PDF, Word DOCX, TXT)")
    print("=" * 60)

    # 1. Load all documents in data/
    print(f"\nLoading all documents from: {data_dir}")
    documents = DocumentLoaderFactory.load_directory(data_dir)
    print(f"[OK] Total documents loaded: {len(documents)}\n")

    for i, doc in enumerate(documents, 1):
        print(f"--- Document #{i}: {doc.title} ---")
        print(f"  - Doc ID     : {doc.doc_id}")
        print(f"  - Source     : {doc.source}")
        print(f"  - Extension  : {doc.metadata.get('extension')}")
        print(f"  - Metadata   : {doc.metadata}")
        print(f"  - Length     : {len(doc.content)} chars")
        print("  - Content Preview (first 200 chars):")
        preview = doc.content[:200].replace("\n", " ")
        print(f"    \"{preview}...\"\n")

    # 2. Test PDF loader with split_pages=True
    pdf_file = data_dir / "sample_ai_intro.pdf"
    if pdf_file.exists():
        print("=" * 60)
        print("Testing PDFLoader with split_pages=True:")
        pdf_loader = PDFLoader(pdf_file, split_pages=True)
        pdf_pages = pdf_loader.load()
        print(f"[OK] PDF split into {len(pdf_pages)} page documents.")
        for page in pdf_pages:
            print(f"  - [{page.doc_id}] Page {page.metadata.get('page_number')}: {len(page.content)} chars")

    # 3. Test JSON Export (Week 1 Data Structure concept)
    print("=" * 60)
    print("Testing Document JSON serialization (Week 1):")
    sample_doc = documents[0]
    print(sample_doc.to_json(indent=2)[:300] + "\n  ...\n}")


if __name__ == "__main__":
    main()
