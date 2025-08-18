from typing import Dict, List

from app.domain.ports.llm import LLMPort


class DummyLLMAdapter(LLMPort):
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return "..."
        return "bot reply"
