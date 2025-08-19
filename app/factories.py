from functools import partial

from fastapi import Depends

from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.openai import OpenAIAdapter
from app.domain.parser import parse_topic_side
from app.repositories.base import get_repo
from app.services.message_service import MessageService
from app.settings import settings


def get_llm(provider: str = None):
    if provider == 'openai':
        return OpenAIAdapter(
            api_key=settings.OPENAI_API_KEY,
            max_history=settings.HISTORY_LIMIT,
        )
    return DummyLLMAdapter()


def get_service(
    repo=Depends(get_repo),
    llm=Depends(partial(
        get_llm,
        provider=settings.LLM_PROVIDER,
        ))
) -> MessageService:
    return MessageService(parser=parse_topic_side, repo=repo, llm=llm)
