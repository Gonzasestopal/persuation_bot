from typing import List

from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class DummyLLMAdapter(LLMPort):
    async def generate(self, conversation: Conversation) -> str:
        return f'I am a bot that defends {conversation.topic} and im side {conversation.side}'

    async def debate(self, messages: List[Message]) -> str:
        return f'After considering your last {len(messages)} messages, this is not totally correct.'
