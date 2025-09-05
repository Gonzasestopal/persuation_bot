from fastapi import Request
from psycopg_pool import AsyncConnectionPool

from app.adapters.repositories.memory import InMemoryMessageRepo
from app.adapters.repositories.pg import PgMessageRepo
from app.settings import settings


def get_pool(request: Request) -> AsyncConnectionPool:
    return request.app.state.dbpool


def get_repo(request: Request) -> PgMessageRepo:
    if settings.USE_INMEMORY_REPO:
        return InMemoryMessageRepo()

    return PgMessageRepo(pool=request.app.state.dbpool)
