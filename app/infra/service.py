from fastapi import Depends

from app.domain.parser import parse_topic_side
from app.domain.ports.llm import LLMPort
from app.infra.db import get_repo
from app.infra.llm import get_llm_singleton
from app.infra.state_store import get_state_store
from app.services.concession_service import ConcessionService
from app.services.message_service import MessageService


def get_concession_singleton(
    store=Depends(get_state_store),
    llm: LLMPort = Depends(get_llm_singleton),
) -> ConcessionService:
    return ConcessionService(llm=llm, state_store=store)


def get_service(
    repo=Depends(get_repo),
    llm: LLMPort = Depends(get_llm_singleton),
    concession: ConcessionService = Depends(get_concession_singleton),
    store=Depends(get_state_store),
) -> MessageService:
    return MessageService(
        parser=parse_topic_side,
        repo=repo,
        llm=llm,
        concession_service=concession,
        state_store=store,
    )
