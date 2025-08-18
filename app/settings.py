from typing import AnyStr

from pydantic import AnyUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: AnyUrl
    HISTORY_LIMIT: int = 5
    EXPIRES_MINUTES: int = 60
    REQUEST_TIMEOUT_S: int = 25
    POSTGRES_USER: AnyStr
    POSTGRES_DB: AnyStr
    POSTGRES_PASSWORD: AnyStr
    POOL_MIN: int = 1
    POOL_MAX: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
