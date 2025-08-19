from typing import AnyStr, Optional

from pydantic import AnyUrl, field_validator
from pydantic_settings import BaseSettings

from app.adapters.llm.constants import OpenAIModels, Provider


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
    OPENAI_API_KEY: str
    LLM_PROVIDER: Optional[Provider] = Provider.OPENAI
    LLM_MODEL: str = OpenAIModels.GPT_4O

    class Config:
        env_file = ".env"

    @field_validator("LLM_PROVIDER", mode="before")
    def allow_blank(cls, v):
        if v == "" or v is None:
            return None
        return v


settings = Settings()
