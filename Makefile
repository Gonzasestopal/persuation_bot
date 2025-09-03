.DEFAULT_GOAL := help

DOCKER_COMPOSE ?= docker compose

files = `find ./service ./tests -name "*.py"`

ifeq ($(OS),Windows_NT)
	VENV_PYTHON=.venv\Scripts\python.exe
	UVICORN=.venv\Scripts\uvicorn.exe
	PYTEST=.venv\Scripts\pytest.exe
	YOYO=.venv\Scripts\yoyo.exe
else
	VENV_PYTHON=.venv/bin/python
	UVICORN=.venv/bin/uvicorn
	PYTEST=.venv/bin/pytest
	YOYO=.venv/bin/yoyo
endif

.PHONY: help install run test down clean dev migrate

help:
	@echo "Available commands:"
	@echo "  make install   - Create venv and install dependencies"
	@echo "  make run       - Start services with Docker Compose"
	@echo "  make test      - Run tests with pytest"
	@echo "  make down      - Stop all running Docker services"
	@echo "  make clean     - Remove venv, containers, caches"
	@echo "  make dev       - Run the FastAPI service in development mode"
	@echo "  make migrate   - Run migrations"

install:
	python -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

migrate:
	$(VENV_PYTHON) -m app.migrate

run:
	@$(DOCKER_COMPOSE) up --build

dev:
	$(UVICORN) app.main:app --reload --port 8000

test:
	python -m pytest --cov=app --cov-report=term-missing

down:
ifeq ($(OS),Windows_NT)
	@if exist docker-compose.yml (docker compose down) else (echo No docker-compose.yml found)
else
	@if [ -f docker-compose.yml ]; then docker compose down; else echo "No docker-compose.yml found"; fi
endif
	@echo "Cleaned up services."

clean: down
ifeq ($(OS),Windows_NT)
	@if exist docker-compose.yml (docker compose down -v) else (echo No docker-compose.yml found)
	@if exist .venv rmdir /s /q .venv
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@for /d /r %%D in (__pycache__) do @if exist "%%D" rmdir /s /q "%%D"
else
	@if [ -f docker-compose.yml ]; then docker compose down -v; else echo "No docker-compose.yml found"; fi
	@rm -rf .venv .pytest_cache __pycache__ || true
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
endif
	@echo "Removed containers, volumes, and caches."



lint:
	ruff check app tests

fmt:
	ruff format app tests
	ruff check --fix app tests

commit_check:
	cz check --rev-range origin/master..HEAD
