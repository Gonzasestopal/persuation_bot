from fastapi import Depends

from app.domain.parser import parse_topic_side
from app.domain.ports.llm import LLMPort
from app.domain.ports.nli import NLIPort
from app.infra.db import get_repo
from app.infra.debate_store import get_state_store
from app.infra.llm import get_llm_singleton
from app.infra.nli import get_nli_singleton
from app.services.concession_service import ConcessionService
from app.services.message_service import MessageService


def get_concession_singleton(
    debate_store=Depends(get_state_store),
    nli: NLIPort = Depends(get_nli_singleton),
    llm: LLMPort = Depends(get_llm_singleton),
) -> ConcessionService:
    return ConcessionService(
        llm=llm,
        debate_store=debate_store,
        nli=nli,
        show_strength_hint_to_user=None,
        hint_via_llm_guidance=True,
    )


def get_service(
    repo=Depends(get_repo),
    llm: LLMPort = Depends(get_llm_singleton),
    concession: ConcessionService = Depends(get_concession_singleton),
    debate_store=Depends(get_state_store),
) -> MessageService:
    return MessageService(
        parser=parse_topic_side,
        repo=repo,
        llm=llm,
        concession_service=concession,
        debate_store=debate_store,
    )
