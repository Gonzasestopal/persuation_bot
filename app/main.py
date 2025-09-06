from contextlib import asynccontextmanager

from fastapi import FastAPI
from psycopg_pool import AsyncConnectionPool

from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.settings import settings


def _pool_is_closed(pool) -> bool:
    # psycopg_pool versions expose either `.closed` or `.is_closed`
    return bool(getattr(pool, 'closed', False) or getattr(pool, 'is_closed', False))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dbpool = None
    app.state.inmem_repo = None

    if settings.USE_INMEMORY_REPO:
        from app.adapters.repositories.memory import InMemoryMessageRepo

        app.state.inmem_repo = InMemoryMessageRepo()

    if not settings.DISABLE_DB_POOL:
        app.state.dbpool = AsyncConnectionPool(
            conninfo=settings.DATABASE_URL.encoded_string(),
            min_size=getattr(settings, 'POOL_MIN', 1),
            max_size=getattr(settings, 'POOL_MAX', 10),
            timeout=10,  # wait at most 5s when borrowing from the pool
            open=True,
        )
    try:
        yield
    finally:
        pool = getattr(app.state, 'dbpool', None)
        if pool is not None:
            await pool.close()


app = FastAPI(lifespan=lifespan)

app.include_router(router)

register_exception_handlers(app)


@app.get('/', tags=['health'])
async def healthcheck():
    return {'Welcome to debate BOT': 'Visit /messages to start conversation'}
