.DEFAULT_GOAL := help

ifeq ($(OS),Windows_NT)
	VENV_PYTHON=.venv\Scripts\python.exe
	UVICORN=.venv\Scripts\uvicorn.exe
	PYTEST=.venv\Scripts\pytest.exe
else
	VENV_PYTHON=.venv/bin/python
	UVICORN=.venv/bin/uvicorn
	PYTEST=.venv/bin/pytest
endif

.PHONY: help install run test down clean

help:
	@echo "Available commands:"
	@echo "  make install   - Create venv and install dependencies"
	@echo "  make run       - Run the FastAPI service with uvicorn"
	@echo "  make test      - Run tests with pytest"
	@echo "  make down      - Stop all running Docker services"
	@echo "  make clean     - Remove venv, containers, caches"

install:
	python -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

run:
	$(UVICORN) app.main:app --reload --port 8000

test:
	$(PYTEST) -q

down:
ifeq ($(OS),Windows_NT)
	@if exist docker-compose.yml (docker compose down) else (echo No docker-compose.yml found)
else
	@if [ -f docker-compose.yml ]; then docker compose down; else echo "No docker-compose.yml found"; fi
endif
	@echo "Cleaned up services."

clean: down
ifeq ($(OS),Windows_NT)
	@if exist .venv rmdir /s /q .venv
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist __pycache__ rmdir /s /q __pycache__
	@$(VENV_PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]" || ver > nul
else
	@[ -d .venv ] && rm -rf .venv || true
	@[ -d .pytest_cache ] && rm -rf .pytest_cache || true
	@[ -d __pycache__ ] && rm -rf __pycache__ || true
	@$(VENV_PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]" || true
endif
	@echo "Cleaned up environment."
