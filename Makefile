# Makefile for Student Academic Regulations & Procedures Assistant (Final Project)
# Supports standard 1-command deployment: make setup, make demo, make eval

.PHONY: setup demo eval clean

setup:
	@echo [SETUP] Installing dependencies with uv...
	uv pip install -e .

demo:
	@echo [DEMO] Launching Terminal Assistant with ReAct Agent Loop...
	uv run python rag_client.py

eval:
	@echo [EVAL] Running 12-testcase benchmark (Calibration + Faithfulness)...
	uv run python tests/evaluate_testset.py

clean:
	@echo [CLEAN] Cleaning cache and temporary indices...
	if exist data\index.json del /f /q data\index.json
