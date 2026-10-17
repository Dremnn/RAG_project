# Makefile for Student Academic Regulations & Procedures Assistant (Final Project)
# Supports standard 1-command deployment: make setup, make demo, make eval

.PHONY: setup demo eval clean

setup:
	@echo [SETUP] Installing dependencies from requirements.txt...
	uv pip install -r requirements.txt

demo:
	@echo [DEMO] Launching Terminal Assistant with ReAct Agent Loop...
	uv run python rag_client.py

eval:
	@echo [EVAL] Running 12-testcase benchmark, CSV log and charts generation...
	uv run python tests/evaluate_testset.py

clean:
	uv run python -c "import pathlib; p = pathlib.Path('data/index.json'); p.unlink(missing_ok=True)"

