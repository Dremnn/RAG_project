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
│   ├── activities.json         # Bảng tra cứu 55 hoạt động CTXH (bóc tách từ CTXH.md)
│   ├── students.json           # Cơ sở dữ liệu thông tin sinh viên
│   └── index.json              # Vector index lưu trữ chunk và metadata
├── logs/                       # Quản lý tập trung toàn bộ log và báo cáo
│   ├── eval/                   # Kết quả thẩm định benchmark (make eval)
│   │   ├── eval_report.json    # Báo cáo đánh giá chi tiết định dạng JSON
│   │   ├── eval_report.csv     # Bảng tổng hợp chi tiết từng câu kiểm thử định dạng CSV
│   │   └── charts/             # 4 đồ thị DPI 300 (Token, Latency, Quality, Dashboard)
│   └── app/                    # Log và báo cáo phiên chạy thực tế (make demo)
│       ├── run_*.jsonl         # Log trace và token usage thực tế theo ngày
│       └── sessions/           # Quản lý lịch sử các phiên tương tác
│           ├── session_*.csv   # Log chi tiết từng câu trong phiên tương tác
│           └── charts/         # Đồ thị phiên tương tác
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
│   ├── evaluate_testset.py     # Script benchmark đánh giá
│   ├── generate_charts.py      # Script tự động sinh đồ thị và dashboard
│   └── test_metadata.py        # Kiểm thử tự động metadata và version filtering
├── Makefile                    # Lệnh chạy nhanh (setup, demo, eval, charts)
├── requirements.txt            # Danh sách thư viện phụ thuộc (pip/uv)
├── rag_client.py               # ReAct Terminal Client
├── rag_server.py               # MCP Knowledge & Action Server
└── ai-usage.md                 # Báo cáo sử dụng trợ lý AI
```
---

## 3. Quy ước Metadata và Đặt tên file tài liệu

### 3.1. Block Metadata (Front-matter) ở đầu tài liệu (.md, .txt, .docx)
Mỗi tài liệu mới đưa vào hệ thống được định nghĩa khối front-matter ở đầu file:
```yaml
---
doc_id: A
title: Quy định CTXH
version: 2023
owner: Phòng Công tác sinh viên
---
(nội dung văn bản bắt đầu từ đây)
```
- `doc_id`: Mã định danh chung cho mọi phiên bản của cùng một văn bản (ví dụ: `A_2021` và `A_2023` đều có `doc_id` là `A`).
- `version`: Số năm hoặc ngày có hiệu lực của văn bản.
- `owner`: Đơn vị ban hành/chủ quản (ví dụ: `Phòng Công tác sinh viên`).
- Với tài liệu `.docx`: Khối block này được đặt ở dòng đầu tiên của tài liệu hoặc thông qua Document Properties (title, author).
- Khối front-matter được tự động cắt khỏi nội dung trước khi chunking và embedding để tránh làm nhiễu kết quả tìm kiếm ngữ nghĩa.

### 3.2. Quy ước đặt tên file mới và Fallback
- Quy ước đặt tên file: `<doc_id>_<năm>.<đuôi>` (Ví dụ: `00_Toan_van_Quy_dinh_CTXH_2023.pdf`, `A_2023.pdf`).
- **Thứ tự ưu tiên trích xuất metadata (`extract_metadata(path)`):**
  1. **Header front-matter (`---` ... `---`)**: Đọc các trường `doc_id`, `title`, `version`, `owner`. Gán `metadata_source = "header"`.
  2. **Regex theo tên file (`<doc_id>_<năm>.<đuôi>`)**: Dành cho PDF không thể chỉnh sửa trực tiếp hoặc file không có header. Gán `metadata_source = "filename"`.
  3. **Fallback**: Không khớp regex thì `version = ""`, `doc_id` lấy tên file không đuôi. Gán `metadata_source = "none"`.

### 3.3. Hiển thị nguồn và Lọc phiên bản
- Kết quả tra cứu `search_ctxh_rules` luôn hiển thị theo chuẩn:
  `[Nguồn: <tên_file> | Phiên bản: <năm> | Chunk ID: <chunk_id>]`
- **Quy tắc phiên bản:** Đối với các chunk cùng `doc_id`, hệ thống tự động ưu tiên giữ lại phiên bản lớn nhất (mới nhất). Nếu người dùng yêu cầu tra cứu một năm cụ thể, hệ thống hỗ trợ tham số `year` tùy chọn trong `search_ctxh_rules(..., year="2023")` để lấy chính xác phiên bản được chỉ định.

---

## 4. Hướng dẫn cài đặt và chạy nhanh (1 lệnh)

### Cài đặt
```bash
make setup
# Hoặc: uv pip install -r requirements.txt
# Hoặc: pip install -r requirements.txt
```

### Chay Demo Terminal Assistant
```bash
make demo
# Hoac: uv run python rag_client.py
```

Các lệnh tiện ích trong terminal:
- `/docs` : Xem danh mục các văn bản quy chế trong cơ sở dữ liệu.
- `/sync` : Quét và đồng bộ ngay các file docx/pdf mới thêm vào `data/`.
- `/eval` : Thẩm định độ trung thực (Faithfulness) của câu vừa trả lời.
- `/clear`: Làm mới lịch sử hội thoại.
- `exit`  : Thoát chương trình.

### Chạy Đánh giá Benchmark (Week 5)
```bash
make eval
# Hoặc: uv run python tests/evaluate_testset.py
```
- **Lệnh duy nhất tự động thực thi toàn bộ:**
  1. Chạy 12 câu kiểm thử (đo Calibration, Faithfulness, Token USAGE).
  2. Xuất báo cáo chi tiết JSON: `logs/eval/eval_report.json`.
  3. Xuất bảng tổng hợp CSV từng câu: `logs/eval/eval_report.csv`.
  4. Tự động vẽ và lưu 4 đồ thị DPI 300 vào `logs/eval/charts/` (`01_token_usage.png`, `02_latency_and_calls.png`, `03_quality_metrics.png`, `04_overall_dashboard.png`) để chèn vào `report.docx`.
