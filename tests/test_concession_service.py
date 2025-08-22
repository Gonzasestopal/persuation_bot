from unittest.mock import Mock

import pytest

from app.domain.consession_policy import DebateState
from app.services.concession_service import ConcessionService


@pytest.mark.asyncio
async def test_analyze_conversation_should_return_debate_state():
    debate_state = DebateState()
    llm = Mock()
    svc = ConcessionService(debate_state=debate_state, llm=llm)
    messages = [Mock(), Mock()]

    out = await svc.analyze_conversation(messages=)

    assert isinstance(out, DebateState)
    assert debate_state.assistant_turns == 1
    assert debate_state.positive_judgements == 1
    assert debate_state.positive_judgements == 1

    assert llm.debate.assert_awaited_once_with(messages=messages)
