from typing import Iterable, List, Optional

from openai import OpenAI

from app.adapters.llm.constants import AWARE_SYSTEM_PROMPT, Difficulty, OpenAIModels
from app.domain.concession_policy import DebateState
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class OpenAIAdapter(LLMPort):
    """
    Adapter that renders the system prompt from authoritative DebateState.
    Backward compatible: `state` is optional in `generate` and `debate`.
    """

    def __init__(
        self,
        api_key: str,
        difficulty: Difficulty = Difficulty.EASY,
        client: Optional[OpenAI] = None,
        model: OpenAIModels = OpenAIModels.GPT_4O,
        temperature: float = 0.3,
        max_output_tokens: int = 80,
    ):
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.difficulty = difficulty

    # ---------- prompt helpers ----------

    def _render_system_prompt(self, state: DebateState) -> str:
        """
        Build the system prompt using fields from DebateState.
        No defaults, no runtime_controls: DebateState must provide all attributes.
        """
        if state is None:
            raise ValueError('DebateState is required, got None')

        debate_status = 'ENDED' if state.match_concluded else 'ONGOING'
        return AWARE_SYSTEM_PROMPT.format(
            STANCE=state.stance,
            DEBATE_STATUS=debate_status,
            TURN_INDEX=state.assistant_turns,
            LANGUAGE=state.lang,
        )

    def _build_user_msg(self, topic: str, side: str) -> str:
        return f"You are debating the topic '{topic}'.\nTake the {side} side.\n\n"

    # ---------- low-level request ----------
    def _request(self, input_msgs: Iterable[dict]) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=list(input_msgs),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return resp.output_text

    # ---------- LLMPort API (backward compatible) ----------
    async def generate(
        self, conversation: Conversation, state: Optional[DebateState] = None
    ) -> str:
        system_prompt = self._render_system_prompt(state)
        print(state, '***')
        user_message = self._build_user_msg(conversation.topic, conversation.side)
        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ]
        return self._request(msgs)

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

    async def debate(
        self, messages: List[Message], state: Optional[DebateState] = None
    ) -> str:
        system_prompt = self._render_system_prompt(state)
        mapped = self._map_history(messages)
        input_msgs = [{'role': 'system', 'content': system_prompt}, *mapped]
        return self._request(input_msgs)
        return self._request(input_msgs)
