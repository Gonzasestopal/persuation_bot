from app.domain.models import Conversation
from app.domain.ports.llm import LLMPort


class DummyLLMAdapter(LLMPort):
    async def generate(self, conversation: Conversation) -> str:
        return f"I am a bot that defends {conversation.topic} and im side {conversation.side}"
