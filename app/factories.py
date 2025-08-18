from fastapi import Depends

from app.adapters.llm.dummy import DummyLLMAdapter
from app.domain.parser import parse_topic_side
from app.repositories.base import get_repo
from app.services.message_service import MessageService


def get_llm():
    return DummyLLMAdapter()


def get_service(repo=Depends(get_repo), llm=Depends(get_llm)) -> MessageService:
    return MessageService(parser=parse_topic_side, repo=repo, llm=llm)
