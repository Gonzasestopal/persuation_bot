from typing import Iterable, List, Optional

from openai import OpenAI

from app.adapters.llm.constants import (MEDIUM_SYSTEM_PROMPT, SYSTEM_PROMPT,
                                        AnthropicModels, Difficulty)
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class AnthropicAdapter(LLMPort):
    def __init__(
        self,
        api_key: str,
        difficulty: Difficulty = Difficulty.EASY,
        client: Optional[OpenAI] = None,
        model: AnthropicModels = AnthropicModels.CLAUDE_35,
        temperature: float = 0.3,
        max_output_tokens: int = 120,
    ):
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.difficulty = difficulty

    @property
    def system_prompt(self):
        if self.difficulty == Difficulty.MEDIUM:
            return MEDIUM_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    async def generate(self, conversation: Conversation):
        pass

    async def debate(self, messages: List[Message]):
        pass
