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
        topic: str,
    ) -> Dict[str, any]:
        stance = Stance(side.upper())  # "PRO" or "CON"

        state = self._get_state(conversation_id)
        mapped = self._map_history(messages)
        last_two_eval = self.judge_last_two_messages(conversation=mapped, stance=stance, topic=topic)

        if self._is_positive_judgment(last_two_eval):
            state.positive_judgements += 1

        if state.maybe_conclude():
            state.match_concluded = True
            return self._build_verdict()

        reply = await self.llm.debate(messages=messages)

        state.assistant_turns += 1

        if state.maybe_conclude():
            state.match_concluded = True
            return self._build_verdict()

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

    def _alignment_and_scores_topic_aware(
        self,
        bot_text: str,
        user_text: str,
        bot_stance: Stance,
        topic: str,
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        Returns:
        align: 'OPPOSITE' | 'SAME' | 'UNKNOWN'  (user vs bot's stance)
        pair_scores: best of {user->bot, bot->user}
        thesis_scores: NLI(user_text, bot_thesis)
        """
        bot_clean = self._drop_questions(bot_text)

        # Pairwise (for strength / concession)
        s_u2b = self.nli.score(user_text, bot_clean)
        s_b2u = self.nli.score(bot_clean, user_text)
        pair_scores = s_u2b if max(s_u2b["entailment"], s_u2b["contradiction"]) >= max(s_b2u["entailment"], s_b2u["contradiction"]) else s_b2u

        # Topic+stance thesis
        thesis = self._bot_thesis(topic, bot_stance)
        thesis_scores = self.nli.score(user_text, thesis)

        ent = thesis_scores["entailment"]
        contr = thesis_scores["contradiction"]

        if contr >= self.contradiction_threshold and contr > ent:
            align = "OPPOSITE"   # user argues against the bot's stance
        elif ent >= self.entailment_threshold and ent > contr:
            align = "SAME"       # user supports the bot's stance
        else:
            align = "UNKNOWN"

        return align, pair_scores, thesis_scores

    def judge_last_two_messages(self, conversation: List[dict], stance: Stance, topic: str) -> Optional[Dict[str, any]]:
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


        align, pair_scores, thesis_scores = self._alignment_and_scores_topic_aware(
            bot_txt, user_txt, stance, topic
        )

        pair_ok = self.nli_confident(pair_scores) and len(user_txt) >= 30
        thesis_ok = self.nli_confident(thesis_scores)

        # Decide alignment
        ent, contr = thesis_scores["entailment"], thesis_scores["contradiction"]

        if contr >= self.contradiction_threshold and contr > ent and thesis_ok:
            align = "OPPOSITE"
            concession = True
            reason = "thesis_opposition"
        elif ent >= self.entailment_threshold and ent > contr:
            align = "SAME"
            concession, reason = False, "same_stance"
        else:
            # fallback: if thesis is neutral, check pairwise contradiction
            if pair_scores["contradiction"] >= self.contradiction_threshold and pair_ok:
                align = "OPPOSITE"
                concession = True
                reason = "pairwise_opposition"
            else:
                align = "UNKNOWN"
                concession, reason = False, "underdetermined"

        return {
            "passed_stance": stance.value,
            "alignment": align,
            "concession": concession,
            "reason": reason,   # still keep primary
            "reasons": [reason] + (
                ["pairwise_opposition"] if reason == "thesis_opposition"
                and pair_scores["contradiction"] >= self.contradiction_threshold else []
            ),
            "scores": pair_scores,
            "thesis_scores": thesis_scores,
            "user_text_sample": user_txt,
            "bot_text_sample": bot_txt,
            "topic": topic,
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

    def _bot_thesis(self, topic: str, bot_stance: Stance) -> str:
        t = topic.strip().rstrip(".")
        if bot_stance == Stance.PRO:
            return f"{t}."
        else:  # bot is CON
            return f"It is not true that {t}."

    def nli_confident(self, scores: Dict[str,float], *, pmin=0.75, margin=0.15) -> bool:
        vals = sorted(scores.values(), reverse=True)
        return vals[0] >= pmin and (vals[0]-vals[1]) >= margin

    def _build_verdict(self) -> str:
        return "On balance, the opposing argument addressed key counters with evidence and causality. I concede the point. Match concluded."
