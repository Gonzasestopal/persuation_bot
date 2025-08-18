from fastapi import Request
from psycopg_pool import AsyncConnectionPool

from app.adapters.repositories.pg import PgMessageRepo


def get_pool(request: Request) -> AsyncConnectionPool:
    return request.app.state.dbpool


def get_repo(request: Request) -> PgMessageRepo:
    return PgMessageRepo(pool=request.app.state.dbpool)
