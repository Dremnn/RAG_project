# Tro ly Quy che & Thu tuc Sinh vien (Student Academic Regulations Assistant)

Du an cuoi ky mon AI Programming / AIPR. He thong tro ly AI ho tro giai dap quy che hoc vu, phan tich bien doi so sanh cac phien ban quy dinh (2021 vs 2024), va truc tiep tiep nhan, xac thuc cac don thu tuc hoc vu cua sinh vien thong qua Model Context Protocol (MCP).

---

## 1. Tinh nang noi bat (Dap ung yeu cau Giang vien)

1. **Tang (a) Tra cuu (Grounding & Calibration):**
   - Su dung Hybrid Search ket hop Vector Search (`jina-embeddings-v3`) va BM25 keyword matching.
   - Luon trich dan nguon van ban va Chunk ID thuc te (`[Nguon: ... | Chunk ID: ...]`).
   - Co kha nang **Calibration (tu choi dung)** voi cac cau hoi ngoai pham vi hoac thieu du lieu quy che (`no_answer`).

2. **Tang (b) Bien doi (Synthesis & Reasoning):**
   - Khong copy nguyen van 1 chunk.
   - So sanh doi chieu mau thuan va su thay doi giua cac phien ban quy che: Ban cu 2021 (het hieu luc) vs Ban hien hanh 2024.
   - Tinh toan thoi han nop don (ngay lam viec), dieu kien tin chi va diem ren luyen de duoc bao luu/rut mon.

3. **Tang (c) Hanh dong (Agent Side Effect + Verification):**
   - Cap cong cu bat buoc: `submit_academic_request` (Write Tool - tao don that vao database JSON) -> goi ngay `get_request_status` (Verify Read Tool - kiem tra trang thai that trong he thong).

4. **Kien truc Ky thuat Chuan muc:**
   - **Custom ReAct Agent Loop**: Tu viet tay bang Python thuan, `MAX_STEPS = 6`, parse tool an toan, khong dung LangChain hay CrewAI.
   - **MCP Server doc lap**: Chay qua `stdio`, expose danh muc cong cu de Client discover dong qua `session.list_tools()`.
   - **Typing chat che**: Toan bo models va schemas deu duoc dinh nghia bang `@dataclass` trong `src/models.py`.
   - **Evaluation (Week 5)**: Testset 12 cau trong `tests/testset.jsonl`, do ti le tu choi dung (Calibration) va do trung thuc (Faithfulness - LLM as a judge), luu ket qua vao `logs/eval_report.json` va ghi log thuc thi vao `logs/run_*.jsonl`.

---

## 2. Cau truc thu muc

```
RAG_project/
├── data/                       # Co so van ban quy che va database don
│   ├── QC_Dao_Tao_2021_v1.docx # Quy che 2021 (cu, het hieu luc)
│   ├── QC_Dao_Tao_2024_v2.docx # Quy che 2024 (hien hanh)
│   ├── QC_Hoan_Thi_2024.docx   # Quy dinh hoan thi 2024
│   ├── QC_Cap_Giay_To_2024.docx# Quy trinh cap bang diem, giay xac nhan
│   └── student_requests.json   # Database ho so don hoc vu
├── logs/                       # Log cac phien chay va bao cao danh gia
│   ├── run_*.jsonl             # Log trace va token usage thuc te
│   └── eval_report.json        # Ket qua benchmark 12 cau test
├── prompts/
│   └── system_agent.txt        # System Prompt tach biet khoi code
├── src/
│   ├── chunking/               # Recursive Chunker
│   ├── evaluation/             # Faithfulness Evaluator (LLM-as-a-judge)
│   ├── loaders/                # Document Loaders (docx, pdf, txt)
│   ├── retrieval/              # VectorStore (BM25 + Dense)
│   └── models.py               # Dataclass schemas
├── tests/
│   ├── testset.jsonl           # Tap 12 cau hoi kiem thu (co no_answer)
│   └── evaluate_testset.py     # Script benchmark danh gia
├── Makefile                    # Lenh chay nhanh (setup, demo, eval)
├── rag_client.py               # ReAct Terminal Client
├── rag_server.py               # MCP Knowledge & Action Server
└── ai-usage.md                 # Bao cao su dung tro ly AI
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
