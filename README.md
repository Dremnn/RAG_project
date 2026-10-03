# RAG Project - Multi-Format Document Ingestion Engine

Hệ thống nạp và xử lý tài liệu đa định dạng (PDF, Microsoft Word DOCX, Plain Text, Markdown) phục vụ cho pipeline RAG, được thiết kế theo chuẩn kiến trúc từ **Week 1 đến Week 7**.

---

## 📁 Cấu trúc thư mục

```text
RAG_project/
├── data/                               # Thư mục chứa tài liệu cần nạp (.pdf, .docx, .txt, .md)
│   ├── sample_ai_intro.pdf             # File PDF mẫu (2 trang)
│   ├── sample_company_policy.docx      # File Word mẫu (có đoạn văn & bảng biểu)
│   └── sample_notes.txt                # File text mẫu
├── src/
│   ├── models.py                       # Dataclass Document & DocumentChunk (W1 & W3)
│   └── loaders/                        # Module đọc tài liệu chuyên dụng
│       ├── base.py                     # BaseLoader (Abstract class)
│       ├── pdf_loader.py               # Trích xuất text, số trang, metadata từ file PDF (pypdf)
│       ├── docx_loader.py              # Đọc đoạn văn và bảng biểu từ file Word (python-docx)
│       ├── text_loader.py              # Đọc .txt, .md với cơ chế tự xử lý encoding
│       └── factory.py                  # DocumentLoaderFactory (tự động nhận diện đuôi file)
├── demo_loader.py                      # File demo chạy thử nghiệm
└── README.md
```

---

## 🚀 Hướng dẫn chạy thử nghiệm

Dự án sử dụng `uv` để quản lý môi trường và thư viện:

```powershell
# Chạy demo kiểm tra nạp toàn bộ thư mục data/
uv run python demo_loader.py
```

---

## 💻 Cách sử dụng trong code Python

### 1. Đọc tự động tất cả tài liệu trong thư mục
```python
from src.loaders import DocumentLoaderFactory

# Load tất cả file .pdf, .docx, .txt, .md trong folder
docs = DocumentLoaderFactory.load_directory("data")

for doc in docs:
    print(f"Title: {doc.title} | Format: {doc.metadata['extension']}")
```

### 2. Đọc một file cụ thể (tự động nhận diện loader)
```python
from src.loaders import DocumentLoaderFactory

# Load file Word
doc_word = DocumentLoaderFactory.load_file("data/sample_company_policy.docx")

# Load file PDF
doc_pdf = DocumentLoaderFactory.load_file("data/sample_ai_intro.pdf")
```

### 3. Tách PDF thành từng trang riêng biệt (rất hữu ích cho Citation trong RAG)
```python
from src.loaders import PDFLoader

loader = PDFLoader("data/sample_ai_intro.pdf", split_pages=True)
pages = loader.load()

for page in pages:
    print(f"Page {page.metadata['page_number']}: {len(page.content)} ký tự")
```
