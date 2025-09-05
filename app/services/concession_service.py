import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from app.adapters.nli.hf_nli import HFNLIProvider
from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
from app.domain.models import Message
from app.domain.nli.config import NLIConfig
from app.domain.nli.scoring import ScoringConfig
from app.domain.ports.debate_store import DebateStorePort
from app.domain.ports.llm import LLMPort
from app.nli.ops import (
    agg_max,
    has_support_either_direction,
    is_contradiction_symmetric,
)
from app.utils.text import (
    drop_questions,
    normalize_spaces,
    round3,
    sanitize_end_markers,
    trunc,
    word_count,
)
from app.verdicts import after_end_message, build_verdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Stance(str, Enum):
    PRO = 'PRO'
    CON = 'CON'


class ConcessionService:
    def __init__(
        self,
        llm: LLMPort,
        nli: Optional[HFNLIProvider] = None,
        nli_config: Optional[NLIConfig] = None,
        scoring: Optional[ScoringConfig] = None,
        state_store: Optional[DebateStorePort] = None,
    ) -> None:
        self.nli_config = nli_config or NLIConfig()
        self.scoring = scoring or ScoringConfig()
        self.nli = nli or HFNLIProvider(model_name=self.nli_config.model_name)
        self.llm = llm
        self.state_store = state_store or InMemoryDebateStore()

    async def analyze_conversation(
        self, messages: List[Message], side: Stance, conversation_id: int, topic: str
    ):
        stance = Stance(side.upper())
        state = self.state_store.get(conversation_id)
        if state is None:
            raise RuntimeError(
                f'DebateState missing for conversation_id={conversation_id}'
            )

        logger.debug(
            '[analyze] conv_id=%s stance=%s topic=%s',
            conversation_id,
            stance.value,
            trunc(topic),
        )

        if state.match_concluded:
            return after_end_message(state=state)

        mapped = [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]
        out = self.judge_last_two_messages(
            conversation=mapped, stance=stance, topic=topic
        )

        if out and out.get('concession'):
            state.positive_judgements += 1
            self.state_store.save(conversation_id=conversation_id, state=state)

        if getattr(state, 'maybe_conclude', lambda: False)():
            state.match_concluded = True
            self.state_store.save(conversation_id=conversation_id, state=state)
            return build_verdict(state=state)

        reply = await self.llm.debate(messages=messages, state=state)
        reply = sanitize_end_markers(reply)

        state.assistant_turns += 1
        if getattr(state, 'maybe_conclude', lambda: False)():
            state.match_concluded = True
            self.state_store.save(conversation_id=conversation_id, state=state)
            return build_verdict(state=state)

        self.state_store.save(conversation_id=conversation_id, state=state)
        return reply.strip()

    # -------- core judge --------
    def _alignment_and_scores_topic_aware(
        self, bot_text: str, user_text: str, bot_stance: Stance, topic: str
    ):
        bot_clean = drop_questions(bot_text)
        user_clean = normalize_spaces(user_text)
        pair_scores = self.nli.bidirectional_scores(bot_clean, user_clean)
        thesis = self._bot_thesis(topic, bot_stance)
        thesis_scores = self.nli.bidirectional_scores(thesis, user_clean)

        if is_contradiction_symmetric(thesis_scores, self.scoring, logger=logger):
            align = 'OPPOSITE'
        else:
            supported, _ = has_support_either_direction(
                thesis_scores, self.scoring, logger=logger
            )
            align = 'SAME' if supported else 'UNKNOWN'

        logger.debug(
            '[align] pair agg=%s | thesis agg=%s',
            round3(agg_max(pair_scores)),
            round3(agg_max(thesis_scores)),
        )
        return align, pair_scores, thesis_scores, thesis

    def _is_on_topic(self, thesis_scores: Dict[str, Dict[str, float]]) -> bool:
        ph = thesis_scores.get('p_to_h', {}) or {}
        hp = thesis_scores.get('h_to_p', {}) or {}

        def has_signal(d):
            ent, contra, neu = (
                float(d.get('entailment', 0.0)),
                float(d.get('contradiction', 0.0)),
                float(d.get('neutral', 1.0)),
            )
            return (max(ent, contra) >= self.scoring.topic_signal_min) or (
                neu <= self.scoring.topic_neu_max
            )

        return has_signal(ph) or has_signal(hp)

    def judge_last_two_messages(
        self, conversation: List[dict], stance: Stance, topic: str
    ) -> Optional[Dict[str, Any]]:
        if not conversation:
            return None
        user_idx = next(
            (
                i
                for i in range(len(conversation) - 1, -1, -1)
                if conversation[i].get('role') == 'user'
            ),
            None,
        )
        if user_idx is None:
            return None
        bot_idx = next(
            (
                i
                for i in range(user_idx - 1, -1, -1)
                if conversation[i].get('role') == 'assistant'
                and word_count(conversation[i].get('content', '')) >= 10
            ),
            None,
        )
        if bot_idx is None:
            return None

        user_txt = conversation[user_idx]['content']
        bot_txt = conversation[bot_idx]['content']
        user_wc = word_count(user_txt)

        align, pair_scores, thesis_scores, thesis = (
            self._alignment_and_scores_topic_aware(bot_txt, user_txt, stance, topic)
        )
        on_topic = self._is_on_topic(thesis_scores)

        thesis_is_contra = is_contradiction_symmetric(
            thesis_scores, self.scoring, logger=logger
        )
        thesis_contra_p = agg_max(thesis_scores).get('contradiction', 0.0)

        if thesis_is_contra and (
            user_wc >= self.scoring.min_user_words
            or thesis_contra_p >= self.scoring.strict_contra_threshold
        ):
            return self._mk_result(
                stance,
                'OPPOSITE',
                True,
                'thesis_opposition',
                pair_scores,
                thesis_scores,
                user_txt,
                bot_txt,
                topic,
            )

        supported, _ = has_support_either_direction(
            thesis_scores, self.scoring, logger=logger
        )
        if supported:
            return self._mk_result(
                stance,
                'SAME',
                False,
                'same_stance',
                pair_scores,
                thesis_scores,
                user_txt,
                bot_txt,
                topic,
            )

        if (
            on_topic
            and is_contradiction_symmetric(pair_scores, self.scoring, logger=logger)
            and user_wc >= self.scoring.min_user_words
        ):
            return self._mk_result(
                stance,
                'OPPOSITE',
                True,
                'pairwise_opposition',
                pair_scores,
                thesis_scores,
                user_txt,
                bot_txt,
                topic,
            )

        if user_wc < self.scoring.min_user_words:
            return self._mk_result(
                stance,
                'UNKNOWN',
                False,
                'too_short',
                pair_scores,
                thesis_scores,
                user_txt,
                bot_txt,
                topic,
            )

        if not on_topic:
            return self._mk_result(
                stance,
                'UNKNOWN',
                False,
                'off_topic',
                pair_scores,
                thesis_scores,
                user_txt,
                bot_txt,
                topic,
            )

        return self._mk_result(
            stance,
            'UNKNOWN',
            False,
            'underdetermined',
            pair_scores,
            thesis_scores,
            user_txt,
            bot_txt,
            topic,
        )

    @staticmethod
    def _mk_result(
        stance,
        alignment,
        concession,
        reason,
        pair_scores,
        thesis_scores,
        user_txt,
        bot_txt,
        topic,
    ):
        return {
            'passed_stance': stance.value,
            'alignment': alignment,
            'concession': concession,
            'reason': reason,
            'reasons': [reason],
            'scores': agg_max(pair_scores),
            'thesis_scores': agg_max(thesis_scores),
            'user_text_sample': user_txt,
            'bot_text_sample': bot_txt,
            'topic': topic,
        }

    @staticmethod
    def _bot_thesis(topic: str, bot_stance: Stance) -> str:
        t = topic.strip().rstrip('.')
        return f'{t}.' if bot_stance == Stance.PRO else f'It is not true that {t}.'
