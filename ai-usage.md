# AI Usage Report (Bao cao su dung Tro ly AI)

## 1. Thong tin du an
- **De tai:** Tro ly Quy che & Thu tuc Sinh vien (Student Academic Regulations & Procedures Assistant)
- **Mon hoc:** AI Programming / Advanced Information Processing & Retrieval (AIPR)
- **Cong nghe:** Python 3.11+, uv, Model Context Protocol (MCP), Groq LLM API (`qwen/qwen3.8-27b`), Jina Embeddings v3 (`jina-embeddings-v3`).

## 2. Muc do va Pham vi su dung Tro ly AI
Trong qua trinh phat trien du an, Tro ly AI duoc su dung nhu mot Pair Programmer ho tro cac cong viec sau:

### 2.1. Thiet ke kien truc & Mo hinh hoa du lieu
- Ho tro thiet ke cac `@dataclass` trong `src/models.py` (`StudentRequest`, `SearchRegulationsInput`, `SubmitRequestInput`, `GetRequestStatusInput`).
- Thao luan va hien thuc kien truc ReAct Agent Loop doc lap, khong phu thuoc framework cao cap (LangChain/CrewAI).

### 2.2. Xay dung bo du lieu van ban mau (Ground-truth Data)
- Tao cac file van ban quy che phien ban hoa de kiem thu tinh nang bien doi va so sanh:
  - `data/QC_Dao_Tao_2021_v1.docx`: Quy che dao tao cu nam 2021 (da het hieu luc).
  - `data/QC_Dao_Tao_2024_v2.docx`: Quy che dao tao hien hanh nam 2024.
  - `data/QC_Hoan_Thi_2024.docx`: Quy dinh hoan thi va thi bo sung 2024.
  - `data/QC_Cap_Giay_To_2024.docx`: Quy trinh cap bang diem va giay xac nhan 2024.
  - `data/student_requests.json`: Co so du lieu JSON luu tru cac don hoc vu thuc te.

### 2.3. Xay dung tap danh gia (Evaluation Testset)
- Ho tro soan thao 12 cau hoi kiem thu trong `tests/testset.jsonl`:
  - 8 cau hoi nghiep vu ve so sanh phien ban, tinh toan han nop va tiep nhan don.
  - 4 cau hoi ngoai pham vi (`no_answer`) de do chi so Calibration (Refusal Rate).

## 3. Phan doan va Kiem soat cua con nguoi (Human Oversight)
- Toan bo ma nguon, logic phan quyet, cac tool schema va thong so thoi gian/dieu kien quy che deu duoc sinh vien kiem tra, tinh chinh va kiem thu thuc te.
- Khong sao chep mu quang ma luon chay danh gia thuc te tren terminal, kiem chung cac file log JSONL (`logs/run_*.jsonl`, `logs/eval_report.json`).
