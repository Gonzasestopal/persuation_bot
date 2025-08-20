from functools import partial
from typing import Optional

from fastapi import Depends

from app.adapters.llm.constants import Difficulty, OpenAIModels, Provider
from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.openai import OpenAIAdapter
from app.domain.exceptions import ConfigError
from app.domain.parser import parse_topic_side
from app.repositories.base import get_repo
from app.services.message_service import MessageService
from app.settings import settings


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    if provider != Provider.OPENAI.value:
        return DummyLLMAdapter()

    if not settings.OPENAI_API_KEY:
        raise ConfigError("OPENAI_API_KEY is required for provider=openai")

    if not settings.DIFFICULTY:
        raise ConfigError("DIFFICULTY is required")

    try:
        difficulty = Difficulty(settings.DIFFICULTY.strip().lower())
    except ValueError:
        raise ConfigError("ONLY EASY AND MEDIUM DIFFICULTY ARE SUPPORTED")

    wanted_model = (model or OpenAIModels.GPT_4O.value).strip().lower()
    try:
        model_enum = OpenAIModels(wanted_model)
    except ValueError:
        raise ConfigError(f"{wanted_model} is not a valid OpenAI model")

    return OpenAIAdapter(
        api_key=settings.OPENAI_API_KEY,
        model=model_enum,
        difficulty=difficulty,
    )

def get_service(
    repo=Depends(get_repo),
    llm=Depends(partial(
        get_llm,
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
    ))
) -> MessageService:
    return MessageService(parser=parse_topic_side, repo=repo, llm=llm)
