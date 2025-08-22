from typing import List

from app.domain.consession_policy import DebateState
from app.domain.models import Message
from app.domain.ports.llm import LLMPort


class ConcessionService:
    def __init__(self, debate_state: DebateState, llm: LLMPort) -> None:
        """
        ConcessionService tracks and analyzes debate state.
        :param debate_state: The DebateState instance that stores current debate info.
        """
        self.debate_state = debate_state
        self.llm = llm

    def analyze_conversation(self, messages: List[Message]) -> DebateState:
        """
        Analyzes the whole conversation.
        Currently returns True unconditionally (placeholder).
        Replace with real logic that inspects debate_state.
        """
        # TODO: Add logic to inspect self.debate_state.messages[-2:]

        return self.debate_state
