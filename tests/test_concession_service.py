from unittest.mock import Mock

from app.domain.consession_policy import DebateState
from app.services.concession_service import ConcessionService


def test_analyze_conversation_should_return_debate_state():
    debate_state = DebateState()
    llm = Mock()
    svc = ConcessionService(debate_state=debate_state, llm=llm)

    out = svc.analyze_conversation(messages=[Mock(), Mock()])

    assert isinstance(out, DebateState)
    assert debate_state.assistant_turns == 1
    assert debate_state.positive_judgements == 1
