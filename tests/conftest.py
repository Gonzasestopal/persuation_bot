# tests/conftest.py
import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
from app.domain.parser import parse_topic_side

# Load env first (OPENAI_API_KEY, etc.)
load_dotenv()

# IMPORTANT: set flags BEFORE importing app/factories/settings so nothing tries to init a DB
os.environ.setdefault('USE_INMEMORY_REPO', '1')
os.environ.setdefault('DISABLE_DB_POOL', '1')

from app.adapters.llm.openai import OpenAIAdapter  # adjust import if different
from app.adapters.repositories.memory import InMemoryMessageRepo
from app.factories import get_service
from app.main import app  # import after flags
from app.services.message_service import MessageService
from app.settings import settings


@pytest.fixture(scope='session')
def client():
    """
    Use a single in-memory repo for the whole test session and the REAL LLM.
    Dependency override guarantees the route won't touch app.state.dbpool.
    """
    # Shared, persistent in-memory repo across requests
    repo = InMemoryMessageRepo()

    # Real LLM adapter (requires OPENAI_API_KEY in env)
    llm = OpenAIAdapter(
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.3,
    )

    state_store = InMemoryDebateStore()

    service = MessageService(
        parser=parse_topic_side, repo=repo, llm=llm, state_store=state_store
    )

    # Override FastAPI DI so routes use our service (no DB access)
    app.dependency_overrides[get_service] = lambda: service

    # Ensure lifespan runs (even though we don't need dbpool)
    with TestClient(app) as c:
        yield c

    # Cleanup override (optional)
    app.dependency_overrides.clear()  # Cleanup override (optional)
    app.dependency_overrides.clear()
    app.dependency_overrides.clear()
