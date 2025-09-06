import abc
from typing import List

from app.domain.concession_policy import DebateState
from app.domain.models import Conversation, Message


class LLMPort(abc.ABC):
    @abc.abstractmethod
    async def generate(self, conversation: Conversation, state: DebateState) -> str:
        """
        Given a conversation topic and side,
        return the assistant's reply as plain text.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def debate(
        self,
        messages: List[Message],
        state: DebateState,
        guidance: str = None,
        responde_mode: str = None,
    ) -> str:
        """
        Given a conversation history (list of Messages),
        return the assistant's reply as plain text.
        """
        raise NotImplementedError
