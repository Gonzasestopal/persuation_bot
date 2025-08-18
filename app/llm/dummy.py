from typing import Dict, List

from app.llm.interface import LLMAdapterInterface


class DummyLLMAdapter(LLMAdapterInterface):
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return "..."
        return "bot reply"
