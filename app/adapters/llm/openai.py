# app/adapters/llm/openai_adapter.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

from openai import OpenAI

from app.adapters.llm.constants import AWARE_SYSTEM_PROMPT, Difficulty, OpenAIModels
from app.domain.concession_policy import DebateState
from app.domain.enums import Stance
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort

STEERING_BLOCK = """# Debate Steering
response_mode: {RESPONSE_MODE}
guidance: {GUIDANCE}
Rules:
- defend: address the strongest objection, provide evidence, never change stance.
- soft_concede: acknowledge one valid point briefly, then reinforce the thesis.
- partial_concede: concede a *specific* sub-claim; delimit scope; restate thesis.
- full_concede: concede fully, end politely, and include <DEBATE_ENDED>.
"""


def _as_stance_str(val: Union[str, Stance, None]) -> str:
    if isinstance(val, Stance):
        return val.value.upper()
    if isinstance(val, str):
        return val.upper()
    return 'PRO'


class OpenAIAdapter(LLMPort):
    """
    Adapter that renders the system prompt from DebateState.
    - Accepts optional `guidance` and `response_mode` to *steer* the reply.
    - Backward compatible: `state` is optional in `debate` (we fall back to safe defaults).
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

    def _render_system_prompt(
        self,
        state: Optional[DebateState],
        *,
        response_mode: Optional[str] = None,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Build the system prompt with DebateState + optional steering block.
        Falls back to safe defaults if `state` is None (keeps things robust).
        """
        stance = _as_stance_str(getattr(state, 'stance', None))
        debate_status = (
            'ENDED' if getattr(state, 'match_concluded', False) else 'ONGOING'
        )
        turn_index = int(getattr(state, 'assistant_turns', 0) or 0)
        topic = (getattr(state, 'topic', None) or '').strip()

        system = AWARE_SYSTEM_PROMPT.format(
            STANCE=stance,
            DEBATE_STATUS=debate_status,
            TURN_INDEX=turn_index,
            LANGUAGE=state.lang or 'auto',
            TOPIC=topic,
        )

        if response_mode or guidance:
            system += '\n' + STEERING_BLOCK.format(
                RESPONSE_MODE=(response_mode or 'defend'),
                GUIDANCE=(guidance or 'Defend the thesis with clear evidence.'),
            )

        return system

    def _build_user_msg(self, topic: str, stance: Stance) -> str:
        return (
            f"You are debating the topic '{topic}'.\nTake the {stance.value} stance.\n"
        )

    @staticmethod
    def _map_history(
        messages: Union[List[Message], List[dict]],
    ) -> List[dict]:
        """
        Accepts domain Message[] or already-mapped [{'role','content'}] and normalizes to the latter.
        """
        if not messages:
            return []
        if isinstance(messages[0], dict):
            # assume already in {'role', 'content'} shape
            return messages  # type: ignore[return-value]
        # domain -> provider
        out: List[dict] = []
        for m in messages:  # type: ignore[assignment]
            role = 'assistant' if getattr(m, 'role', '') == 'bot' else 'user'
            out.append({'role': role, 'content': getattr(m, 'message', '')})
        return out

    # ---------- low-level request ----------

    def _request(self, input_msgs: Sequence[dict]) -> str:
        """
        Uses Responses API for consistency with your earlier code.
        """
        resp = self.client.responses.create(
            model=self.model,  # OpenAI enum -> string
            input=list(input_msgs),  # [{'role', 'content'}, ...]
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return resp.output_text

    # ---------- public API ----------

    async def generate(self, conversation: Conversation, state: DebateState) -> str:
        """
        First turn for a new conversation (system + user kickoff).
        """
        system_prompt = self._render_system_prompt(state)
        user_message = self._build_user_msg(conversation.topic, conversation.stance)

        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ]
        return self._request(msgs)

    async def debate(
        self,
        *,
        messages: Union[List[Message], List[dict]],
        state: Optional[DebateState] = None,
        guidance: Optional[str] = None,
        response_mode: Optional[str] = None,
    ) -> str:
        """
        Continuation turns. `guidance` + `response_mode` are optional steering signals
        coming from ConcessionService's policy tier.
        """
        system_prompt = self._render_system_prompt(
            state, response_mode=response_mode, guidance=guidance
        )
        mapped = self._map_history(messages)
        input_msgs = [{'role': 'system', 'content': system_prompt}, *mapped]
        return self._request(input_msgs)
