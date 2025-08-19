from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool

from app.api.messages import router
from app.domain.exceptions import ConfigError
from app.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dbpool = AsyncConnectionPool(
        conninfo=settings.DATABASE_URL.encoded_string(),
        min_size=settings.POOL_MIN,
        max_size=settings.POOL_MAX,
        open=True,
    )
    try:
        yield
    finally:
        await app.state.dbpool.close()
        await app.state.dbpool.wait_closed()

app = FastAPI(lifespan=lifespan)

app.include_router(router)


@app.exception_handler(ConfigError)
async def config_error_handler(_: Request, exc: ConfigError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
