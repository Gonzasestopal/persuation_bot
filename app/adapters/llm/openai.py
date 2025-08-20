from typing import Iterable, List, Optional

from openai import OpenAI

from app.adapters.llm.constants import SYSTEM_PROMPT, OpenAIModels
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
        max_output_tokens: int = 80,
    ):
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_history = max_history
        self.max_output_tokens = max_output_tokens

    @property
    def system_prompt(self):
        return SYSTEM_PROMPT

    def _build_user_msg(self, topic: str, side: str):
        return (
            f"You are debating the topic '{topic}'.\n"
            f"Take the {side} side.\n\n"
        )

    def _request(self, input_msgs: Iterable[dict]) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=list(input_msgs),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return resp.output_text

    async def generate(self, conversation: Conversation) -> str:
        user_message = self._build_user_msg(conversation.topic, conversation.side)
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self._request(msgs)

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [{"role": ("assistant" if m.role == "bot" else "user"), "content": m.message}
                for m in messages]

    async def debate(self, messages: List[Message]) -> str:
        mapped = self._map_history(messages)
        input_msgs = [{"role": "system", "content": self.system_prompt}]
        input_msgs.extend(mapped)
        return self._request(input_msgs)
