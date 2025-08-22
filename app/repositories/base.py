from fastapi import Request
from psycopg_pool import AsyncConnectionPool

from app.adapters.repositories.memory import InMemoryMessageRepo
from app.adapters.repositories.pg import PgMessageRepo
from app.settings import settings


def get_pool(request: Request) -> AsyncConnectionPool:
    return request.app.state.dbpool


def get_repo(request: Request) -> PgMessageRepo:
    if settings.USE_INMEMORY_REPO:
        # Reuse the single instance created in lifespan
        repo = getattr(request.app.state, "inmem_repo", None)
        if repo is None:
            repo = request.app.state.inmem_repo = InMemoryMessageRepo()

    return PgMessageRepo(pool=request.app.state.dbpool)
