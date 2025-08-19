from typing import List, Optional

from openai import OpenAI

from app.adapters.llm.constants import OpenAIModels
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class OpenAIAdapter(LLMPort):
    def __init__(
        self,
        api_key: str,
        max_history: int,
        client: Optional[OpenAI] = None,
        model: OpenAIModels = OpenAIModels.GPT_4O,
        temperature: float = 0.3,
    ):
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_history = max_history

    async def generate(self, conversation: Conversation) -> str:
        raise NotImplementedError

    async def debate(self, messages: List[Message]) -> str:
        raise NotImplementedError
