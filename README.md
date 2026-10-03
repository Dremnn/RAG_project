# RAG Knowledge Assistant - Architecture W1 to W7

Hệ thống RAG Agent thông minh chạy trên Terminal, được thiết kế theo chuẩn kiến trúc từ **Week 1 đến Week 7** với sự kết hợp của **Model Context Protocol (MCP)**, **Hybrid Search (BM25 + Dense Vector)**, **Smart Incremental Sync** và **LLM-as-a-Judge Faithfulness Evaluation**.

---

## 🏛️ Sơ đồ Kiến trúc Tổng thể (Architecture Map)

```text
┌────────────────────────────────────────────────────────┐
│            TERMINAL MCP CLIENT (rag_client.py)         │
│  - Giao diện dòng lệnh tương tác trực tiếp             │
│  - ReAct Agent Loop (W6): suy luận & gọi tool          │
│  - Strict Grounding & Anti-Hallucination               │
│  - LLM-as-a-judge Faithfulness Evaluator (W5)          │
└──────────────────────────┬─────────────────────────────┘
                           │ (Giao thức chuẩn MCP / stdio)
                           ▼
┌────────────────────────────────────────────────────────┐
│             RAG MCP SERVER (rag_server.py)             │
│  - Multi-format Loaders: PDF, DOCX, TXT, MD (W1 & W3)  │
│  - Recursive Character Chunking with Overlap (W3)      │
│  - Unified VectorStore with Hybrid Search (W3 & W4)    │
│  - Smart Incremental Sync (Tự động nhận diện file mới) │
│  - Khai báo MCP Tools:                                 │
│      • search_documents(query, top_k)                  │
│      • list_indexed_documents()                        │
│      • get_document_full(doc_id)                       │
│      • sync_documents()                                │
└────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc Thư mục

```text
RAG_project/
├── data/                               # Kho tài liệu nội bộ (.pdf, .docx, .txt, .md)
│   ├── sample_ai_intro.pdf
│   ├── sample_company_policy.docx
│   ├── sample_notes.txt
│   └── index.json                      # Cache Vector và Metadata đã lập chỉ mục
├── src/
│   ├── models.py                       # Dataclass Document, DocumentChunk (W1 & W3)
│   ├── loaders/                        # Module đọc tài liệu đa định dạng (W1 & W3)
│   │   ├── base.py
│   │   ├── pdf_loader.py               # pypdf (hỗ trợ phân trang)
│   │   ├── docx_loader.py              # python-docx (bóc tách bảng biểu sang Markdown)
│   │   ├── text_loader.py              # Xử lý text/markdown/encoding
│   │   └── factory.py                  # DocumentLoaderFactory tự động phân loại
│   ├── chunking/                       # Cắt nhỏ văn bản (W3)
│   │   └── splitter.py                 # RecursiveChunker (ưu tiên đoạn > câu > từ + overlap)
│   ├── retrieval/                      # Động cơ tìm kiếm (W3 & W4)
│   │   ├── embeddings.py               # Jina AI Embeddings v3 (1024 chiều)
│   │   └── vector_store.py             # VectorStore tích hợp Hybrid Search (BM25 + Cosine)
│   └── evaluation/                     # Thẩm định độ tin cậy (W5)
│       └── faithfulness.py             # Phân rã Atomic Claims & LLM-as-a-judge
├── rag_server.py                       # MCP Server chuẩn Week 7
├── rag_client.py                       # Terminal Client & ReAct Agent (Week 6 & Week 7)
├── demo_loader.py                      # Test độc lập Loaders
├── demo_chunker.py                     # Test độc lập Chunking
├── demo_hybrid_search.py               # Test độc lập Hybrid Search
├── demo_evaluation.py                  # Test độc lập Faithfulness Evaluation
└── README.md
```

---

## 🚀 Hướng dẫn Sử dụng

### 1. Khởi chạy Trợ lý RAG trên Terminal
```powershell
cd "RAG_project"
uv run .\rag_client.py
```

### 2. Các lệnh tiện ích trong Terminal
* **Hỏi đáp tự nhiên:** Nhập bất kỳ câu hỏi nào về nội dung các tài liệu trong `data/`.
* **`/docs`**: Xem danh mục toàn bộ file hiện có trong cơ sở tri thức.
* **`/sync`**: Quét và nạp ngay các file PDF, Word mới bạn vừa thả vào `data/` mà không cần thoát Terminal.
* **`/eval`**: Thẩm định độ trung thực (Faithfulness) của câu trả lời vừa rồi để phát hiện ảo giác (Hallucination).
* **`/clear`**: Xóa ngữ cảnh để bắt đầu phiên hỏi đáp mới.
* **`exit`**: Thoát chương trình.
