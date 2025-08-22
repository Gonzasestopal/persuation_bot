from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool

from app.api.routes import router
from app.domain.exceptions import ConfigError
from app.settings import settings


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
            min_size=settings.POOL_MIN,
            max_size=settings.POOL_MAX,
            open=True,
        )
    try:
        yield
    finally:
        pool = getattr(app.state, "dbpool", None)
        if pool is not None:
            await pool.close()
            await pool.wait_closed()

app = FastAPI(lifespan=lifespan)

app.include_router(router)


@app.exception_handler(ConfigError)
async def config_error_handler(_: Request, exc: ConfigError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/", tags=["health"])
async def healthcheck():
    return {"Welcome to debate BOT": "Visit /messages to start conversation"}
