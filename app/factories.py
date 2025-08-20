from functools import partial
from typing import Optional

from fastapi import Depends

from app.adapters.llm.constants import OpenAIModels, Provider
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
    if provider == Provider.OPENAI.value:
        if not settings.OPENAI_API_KEY:
            raise ConfigError("OPENAI_API_KEY is required for provider=openai")
        default_model = model or OpenAIModels.GPT_4O.value
        try:
            model = OpenAIModels(default_model).value
        except ValueError:
            raise ConfigError(f"{default_model} is not a valid OpenAI model")

        max_history_messages = None if settings.HISTORY_LIMIT == 0 else settings.HISTORY_LIMIT * 2

        return OpenAIAdapter(
            api_key=settings.OPENAI_API_KEY,
            max_history=max_history_messages,
            model=default_model,
        )
    return DummyLLMAdapter()


def get_service(
    repo=Depends(get_repo),
    llm=Depends(partial(
        get_llm,
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
    ))
) -> MessageService:
    return MessageService(parser=parse_topic_side, repo=repo, llm=llm)
