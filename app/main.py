from contextlib import asynccontextmanager

from fastapi import FastAPI
from psycopg_pool import AsyncConnectionPool

from app.api.messages import router
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


@app.get("/")
def read_root():
    return {"Hello": "World"}
