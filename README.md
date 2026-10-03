# Mini RAG Assistant

Hệ thống hỏi đáp tài liệu thông minh (RAG) chạy trên Terminal, được xây dựng theo chuẩn kiến trúc **Model Context Protocol (MCP)** và tích hợp từ Week 1 đến Week 7.

---

## 🌟 Tính năng nổi bật

- **Đọc đa định dạng**: Tự động đọc và xử lý file PDF (có phân trang), Word DOCX (trích xuất văn bản & bảng biểu), Text, Markdown.
- **Hybrid Search**: Kết hợp tìm kiếm từ khóa chính xác (**BM25**) và tìm kiếm ngữ nghĩa (**Dense Vector Cosine Similarity** qua Jina Embeddings 1024-dim).
- **Kiến trúc MCP (Week 7)**: Tách biệt **RAG Server** (`rag_server.py`) và **Terminal Client** (`rag_client.py`) qua giao thức `stdio`.
- **ReAct Agent (Week 6)**: Sử dụng mô hình Groq (`qwen/qwen3.8-27b`) tự suy luận, gọi công cụ tìm kiếm và đính kèm trích dẫn nguồn (**Citations**).
- **Smart Incremental Sync**: Tự động phát hiện và nạp các file mới được thêm vào thư mục `data/` mà không cần tính toán lại các file cũ.
- **Thẩm định độ trung thực (Week 5)**: Bẻ nhỏ câu trả lời thành từng mệnh đề thực tế (*Atomic Claims*) và dùng LLM-as-a-judge chấm điểm **Faithfulness** để ngăn ngừa ảo giác (*Hallucination*).

---

## 🛠️ Cài đặt & Chuẩn bị

1. **Yêu cầu môi trường**: Python 3.11+ và công cụ `uv`.
2. **Cấu hình API Key**: Tạo file `.env` tại thư mục gốc với nội dung:
   ```env
   GROQ_API_KEY=your_groq_api_key
   JINA_API_KEY=your_jina_api_key
   ```
3. **Tài liệu**: Thả các file tài liệu (`.pdf`, `.docx`, `.txt`, `.md`) cần tra cứu vào thư mục `data/`.

---

## 🚀 Hướng dẫn Sử dụng

Khởi chạy hệ thống hỏi đáp trực tiếp trên Terminal:

```powershell
uv run python rag_client.py
```

### Các lệnh tiện ích trong phiên chat:
- **Nhập câu hỏi**: Chat tự nhiên để hỏi đáp về nội dung tài liệu.
- **`/docs`**: Xem danh mục toàn bộ tài liệu đang có trong hệ thống.
- **`/sync`**: Quét và nạp ngay tài liệu mới vừa thả vào `data/` mà không cần khởi động lại.
- **`/eval`**: Thẩm định độ trung thực (Faithfulness) của câu trả lời vừa nhận.
- **`/clear`**: Làm mới lịch sử trò chuyện.
- **`exit`**: Thoát chương trình.
