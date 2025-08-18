import abc
from typing import Dict, List


class LLMAdapterInterface(abc.ABC):
    @abc.abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Given a conversation history (list of {role, message} dicts),
        return the assistant's reply as plain text.
        """
        raise NotImplementedError
