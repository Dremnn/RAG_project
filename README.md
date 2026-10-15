# Trợ lý Công tác Xã hội (CTXH Assistant) — Đề tài T5 (HCMUTE)

Dự án cuối kỳ môn AI Programming / AIPR. Hệ thống Trợ lý AI hỗ trợ giải đáp quy định Công tác xã hội (CTXH), tra cứu mã hoạt động 6 chữ số, tính toán quy đổi điểm thành ngày CTXH, thẩm định tiến độ tốt nghiệp, và trực tiếp ghi nhận/đăng ký hoạt động cho sinh viên thông qua Model Context Protocol (MCP).

---

## 1. Tính năng nổi bật (Đáp ứng trọn vẹn 3 tầng bắt buộc)

1. **Tầng (a) Tra cứu (Grounding & Calibration):**
   - Sử dụng **Hybrid Search** kết hợp Dense Vector Search (`jina-embeddings-v3`) và BM25 keyword retrieval.
   - Luôn trích dẫn nguồn văn bản và Chunk ID thực tế (`[Nguồn: CTXH.md | Chunk ID: ...]`).
   - Có cơ chế **Calibration (từ chối đúng)** với các câu hỏi ngoài phạm vi CTXH hoặc câu hỏi từ trường khác (`no_answer`).

2. **Tầng (b) Biến đổi (Synthesis & Calculation):**
   - Không copy nguyên văn 1 chunk mà tổng hợp, suy luận logic.
   - **Tra mã & Quy đổi ngày CTXH**: Mã hóa 6 chữ số (001101, 001202, 002101...), công thức chia 10 và làm tròn theo quy định.
   - **So sánh chính sách theo khóa**: So sánh yêu cầu ngày CTXH tối thiểu giữa khóa 2012 (3 ngày) và từ khóa 2013 trở đi (4 ngày).
   - **Tính toán tiến độ tốt nghiệp**: Đối chiếu số ngày đã tích lũy với chuẩn tối thiểu.
   - **Nhận diện thiếu dữ liệu**: Yêu cầu người dùng bổ sung MSSV khi thiếu thông tin đăng ký.

3. **Tầng (c) Hành động (Side Effect + Verify):**
   - Cặp công cụ bắt buộc: `register_ctxh_activity` (Write Tool — ghi nhận hoạt động và cộng ngày vào CSDL `student_ctxh.json`) → gọi ngay `get_ctxh_status` (Verify Read Tool — đọc lại CSDL thực tế để kiểm chứng trạng thái).

4. **Kiến trúc Kỹ thuật:**
   - **Custom ReAct Agent Loop**: Tự viết tay bằng Python thuần, `MAX_STEPS = 6`, parse tool an toàn, kiểm soát vòng lặp.
   - **MCP Server độc lập**: Giao tiếp qua `stdio`, expose công cụ để Client discover động qua `session.list_tools()`.
   - **Typing chặt chẽ**: Toàn bộ model & schema được định nghĩa bằng `@dataclass` trong `src/models.py`.
   - **Evaluation (Week 5)**: Testset 12 câu trong `tests/testset.jsonl` (gồm 4 câu `no_answer`), đo tỷ lệ từ chối đúng (Calibration), độ trung thực (Faithfulness) và keyword recall, xuất báo cáo JSON & CSV chi tiết.

---

## 2. Cấu trúc thư mục

```
RAG_project/
├── data/                       # Cơ sở văn bản quy định và database sinh viên
│   ├── CTXH.md                 # Quy định Công tác xã hội toàn văn (HCMUTE)
│   ├── student_ctxh.json       # Cơ sở dữ liệu hồ sơ CTXH của sinh viên
│   └── index.json              # Vector index lưu trữ chunk và metadata
├── logs/                       # Log các phiên chạy và báo cáo đánh giá
│   ├── run_*.jsonl             # Log trace và token usage thực tế
│   ├── eval_report.json        # Báo cáo đánh giá chi tiết định dạng JSON
│   └── eval_report.csv         # Bảng tổng hợp chi tiết từng câu kiểm thử định dạng CSV
├── prompts/
│   └── system_agent.txt        # System Prompt 3 tầng tách biệt khỏi code
├── src/
│   ├── chunking/               # Recursive Chunker
│   ├── evaluation/             # Faithfulness Evaluator (LLM-as-a-judge)
│   ├── loaders/                # Document Loaders (docx, pdf, txt, md)
│   ├── retrieval/              # VectorStore (BM25 + Dense Jina v3)
│   └── models.py               # Dataclass schemas
├── tests/
│   ├── testset.jsonl           # Tập 12 câu hỏi kiểm thử (co 4 câu no_answer)
│   └── evaluate_testset.py     # Script benchmark đánh giá
├── Makefile                    # Lệnh chạy nhanh (setup, demo, eval)
├── rag_client.py               # ReAct Terminal Client
├── rag_server.py               # MCP Knowledge & Action Server
└── ai-usage.md                 # Báo cáo sử dụng trợ lý AI
```

---

## 3. Huong dan cai dat va chay nhanh (1 lenh)

### Cai dat
```bash
make setup
# Hoac: uv pip install -e .
```

### Chay Demo Terminal Assistant
```bash
make demo
# Hoac: uv run python rag_client.py
```

Cac lenh tien ich trong terminal:
- `/docs` : Xem danh muc cac van ban quy che trong co so du lieu.
- `/sync` : Quet va dong bo ngay cac file docx/pdf moi them vao `data/`.
- `/eval` : Tham dinh do trung thuc (Faithfulness) cua cau vua tra loi.
- `/clear`: Lam moi lich su hoi thoai.
- `exit`  : Thoat chuong trinh.

### Chay Danh gia Benchmark (Week 5)
```bash
make eval
# Hoac: uv run python tests/evaluate_testset.py
```
Ket qua danh gia tong the va tung cau test se duoc hien thi tren terminal va xuat ra `logs/eval_report.json`.
