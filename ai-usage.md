# AI Usage Report (Bao cao su dung Tro ly AI)

## 1. Thong tin du an
- **De tai:** T5 - Tro ly Cong tac Xa hoi (CTXH): Quy doi & Dang ky (CTXH Assistant)
- **Mon hoc:** AI Programming / Advanced Information Processing & Retrieval (AIPR)
- **Cong nghe:** Python 3.11+, uv, Model Context Protocol (MCP), Groq LLM API (`qwen/qwen3.8-27b`), Jina Embeddings v3 (`jina-embeddings-v3`).

## 2. Muc do va Pham vi su dung Tro ly AI
Trong qua trinh phat trien du an, Tro ly AI duoc su dung nhu mot Pair Programmer ho tro cac cong viec sau:

### 2.1. Thiet ke kien truc & Mo hinh hoa du lieu
- Ho tro thiet ke cac `@dataclass` trong `src/models.py` (`StudentCTXHRecord`, `CTXHRegistration`, `SearchCTXHRulesInput`, `RegisterCTXHInput`, `GetCTXHStatusInput`).
- Thao luan va hien thuc kien truc ReAct Agent Loop doc lap, khong phu thuoc framework cao cap (LangChain/CrewAI).

### 2.2. Xay dung bo du lieu van ban mau (Ground-truth Data)
- Tao file van ban quy dinh CTXH de lam co so tri thuc cho RAG:
  - `data/CTXH.md`: Quy dinh to chuc thuc hien chuong trinh Cong tac xa hoi doi voi sinh vien he chinh quy (HCMUTE). Bao gom: quy dinh chung, ma hoa & quy doi ngay CTXH, quy trinh to chuc & danh gia.
  - `data/student_ctxh.json`: Co so du lieu JSON luu tru ho so CTXH cua sinh vien (MSSV, hoat dong da dang ky, so ngay tich luy).

### 2.3. Xay dung tap danh gia (Evaluation Testset)
- Ho tro soan thao 12 cau hoi kiem thu trong `tests/testset.jsonl`:
  - 3 cau tra cuu quy dinh CTXH (logic_rules): ma hoat dong, dieu kien tot nghiep, xu ly vi pham.
  - 1 cau so sanh phien ban (version_comparison): khoa 2012 vs khoa tu 2013.
  - 2 cau tinh toan quy doi (calculation): tong hop nhieu hoat dong, quy doi diem thanh ngay.
  - 1 cau hanh dong ghi + kiem chung (action_write_verify): dang ky CTXH va verify.
  - 1 cau thieu thong tin (missing_info): yeu cau bo sung MSSV.
  - 4 cau ngoai pham vi (`no_answer`) de do chi so Calibration (Refusal Rate).

### 2.4. Ho tro debug va toi uu hoa
- Ho tro phat hien va sua loi syntax (thut le sai, thieu dau ngoac) trong `rag_server.py`.
- Ho tro toi uu hoa cau hinh Hybrid Search (trong so BM25 vs Dense).

## 3. Phan doan va Kiem soat cua con nguoi (Human Oversight)
- Toan bo ma nguon, logic phan quyet, cac tool schema va thong so thoi gian/dieu kien quy che deu duoc sinh vien kiem tra, tinh chinh va kiem thu thuc te.
- Khong sao chep mu quang ma luon chay danh gia thuc te tren terminal, kiem chung cac file log JSONL (`logs/run_*.jsonl`, `logs/eval_report.json`).
