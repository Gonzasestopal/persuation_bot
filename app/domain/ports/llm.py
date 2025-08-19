import abc

from app.domain.models import Conversation


class LLMPort(abc.ABC):
    @abc.abstractmethod
    async def generate(self, conversation: Conversation) -> str:
        """
        Given a conversation topic and side,
        return the assistant's reply as plain text.
        """
        raise NotImplementedError
