import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.concession_policy import DebateState
from app.domain.models import Message
from app.domain.ports.llm import LLMPort

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # adjust to INFO in production


# --- data classes ---
@dataclass(frozen=True)
class _NLIConfig:
    model_name: str = "roberta-large-mnli"
    entailment_threshold: float = 0.65
    contradiction_threshold: float = 0.70
    max_length: int = 512
    max_claims_per_turn: int = 3


class Stance(str, Enum):
    PRO = "PRO"
    CON = "CON"


SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+')
IS_QUESTION_RX = re.compile(r'\?\s*$')


class ConcessionService:
    def __init__(
        self,
        llm: LLMPort,
        nli: Optional[HFNLIProvider] = None,
        config: _NLIConfig = _NLIConfig(),

    ) -> None:
        """
        ConcessionService tracks and analyzes debate state.
        :param debate_state: The DebateState instance that stores current debate info.
        """
        self._state: Dict[int, DebateState] = {}  # conversation_id -> DebateState
        self.llm = llm
        self.nli = nli or HFNLIProvider()
        self.config = config
        self.entailment_threshold = config.entailment_threshold
        self.contradiction_threshold = config.contradiction_threshold

    async def analyze_conversation(
        self, messages: List[Message],
        side: Stance,
        conversation_id: int,
    ) -> Dict[str, any]:
        stance = Stance(side.upper())  # "PRO" or "CON"

        state = self._get_state(conversation_id)
        mapped = self._map_history(messages)
        last_two_eval = self.judge_last_two_messages(conversation=mapped, stance=stance)

        print(last_two_eval)

        if self._is_positive_judgment(last_two_eval):
            state.positive_judgements += 1

        if state.maybe_conclude():
            state.match_concluded = True
            return "Match concluded."

        reply = await self.llm.debate(messages=messages)

        state.assistant_turns += 1

        print(state.assistant_turns, state.positive_judgements)

        if state.maybe_conclude():
            state.match_concluded = True
            return reply.strip() + "\n\nMatch concluded."

        return reply.strip()

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [{"role": ("assistant" if m.role == "bot" else "user"), "content": m.message} for m in messages]

    def _get_state(self, conversation_id: int) -> DebateState:
        st = self._state.setdefault(conversation_id, DebateState())
        return st

    def _drop_questions(self, text: str) -> str:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        sents = [s for s in sents if not re.search(r'\?\s*$', s)]
        return " ".join(sents) if sents else text

    def _alignment_and_scores(self, stance: Stance, bot_text: str, user_text: str) -> Tuple[str, Dict[str, float]]:
        bot_clean = self._drop_questions(bot_text)
        scores_u2b = self.nli.score(user_text, bot_clean)
        scores_b2u = self.nli.score(bot_clean, user_text)
        chosen = scores_u2b if max(scores_u2b["entailment"], scores_u2b["contradiction"]) >= max(scores_b2u["entailment"], scores_b2u["contradiction"]) else scores_b2u
        ent, contr = chosen["entailment"], chosen["contradiction"]
        if contr >= self.contradiction_threshold and contr > ent:
            align = "OPPOSITE"
        elif ent >= self.entailment_threshold and ent > contr:
            align = "SAME"
        else:
            align = "UNKNOWN"
        return align, chosen

    def judge_last_two_messages(self, conversation: List[dict], stance: Stance) -> Optional[Dict[str, any]]:
        if not conversation:
            return None
        # latest user
        user_idx = self._latest_idx(conversation, "user")
        if user_idx is None:
            return None
        # latest valid assistant before user
        bot_idx = self._latest_valid_assistant_before(conversation, user_idx)
        if bot_idx is None:
            return None
        user_txt = conversation[user_idx]["content"]
        bot_txt  = conversation[bot_idx]["content"]

        align, scores = self._alignment_and_scores(stance, bot_txt, user_txt)

        if align == "OPPOSITE":
            concession = scores["contradiction"] >= self.contradiction_threshold
            reason = "strong_opposition" if concession else "weak_opposition"
        elif align == "SAME":
            concession, reason = False, "same_stance"
        else:
            concession, reason = False, "underdetermined"

        return {
            "passed_stance": stance.value,
            "alignment": align,
            "concession": concession,
            "reason": reason,
            "scores": scores,
            "user_text_sample": user_txt,
            "bot_text_sample": bot_txt,
        }

    def _latest_idx(self, conv: List[dict], role: str, *, before_idx: Optional[int] = None) -> Optional[int]:
        for i in range(len(conv) - 1, -1, -1):
            if before_idx is not None and i >= before_idx:
                continue
            if conv[i].get("role") == role:
                return i
        return None

    def _latest_valid_assistant_before(self, conv: List[dict], before_idx: int, *, min_words: int = 10) -> Optional[int]:
        for i in range(before_idx - 1, -1, -1):
            m = conv[i]
            if m.get("role") != "assistant":
                continue
            words = [w for w in m.get("content","").split() if w.isalpha()]
            if len(words) >= min_words:
                return i
        return None

    def _is_positive_judgment(self, eval_payload: Optional[dict]) -> bool:
        # Keep this definition tight; don’t compare to stance.value.
        return bool(eval_payload and eval_payload.get("concession"))
