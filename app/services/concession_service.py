import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.concession_policy import DebateState
from app.domain.models import Message
from app.domain.ports.llm import LLMPort

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # set to INFO in production

# ----------------------------- Config & Enums -------------------------------


@dataclass(frozen=True)
class _NLIConfig:
    model_name: str = 'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli'
    entailment_threshold: float = 0.65
    contradiction_threshold: float = 0.70
    max_length: int = 512
    max_claims_per_turn: int = 3
    min_user_words: int = 8  # piso de palabras del usuario
    strict_contra_threshold: float = 0.90  # contradicción “extra fuerte”


WORD_RX = re.compile(r'[^\W\d_]+', flags=re.UNICODE)  # palabras alfabéticas


class Stance(str, Enum):
    PRO = 'PRO'
    CON = 'CON'


# ----------------------------- Heurísticas ----------------------------------

# márgenes alineados con tus tests
EPS_CONTRA_VS_NEU = 0.03
EPS_ENT = 0.20
MARGIN_EC = 0.02
MIN_ENT = 0.25

SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+')
IS_QUESTION_RX = re.compile(r'\?\s*$')

# on-topic
TOPIC_NEU_MAX = 0.75  # si neutral > 0.75 en ambas direcciones, probablemente off-topic
TOPIC_SIGNAL_MIN = 0.30  # exigimos algo de señal (ent o contra) en alguna dirección


def _trunc(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + '…'


def _round3(d: dict) -> dict:
    return {
        k: round(float(d.get(k, 0.0)), 3)
        for k in ('entailment', 'neutral', 'contradiction')
    }


def _print(*args, **kwargs):
    print(*args, **kwargs)


# -------------------------------- Service ----------------------------------


class ConcessionService:
    """
    Usa NLI para decidir concesiones:
      - CONTRADICTION se decide simétricamente (agg de ambas direcciones).
      - SUPPORT (entailment) es direccional (mejor de dos), pero NO cuenta para concesión.
    """

    def __init__(
        self,
        llm: LLMPort,
        nli: Optional[HFNLIProvider] = None,
        config: _NLIConfig = _NLIConfig(),
    ) -> None:
        self._state: Dict[int, DebateState] = {}  # conversation_id -> DebateState
        self.llm = llm
        self.nli = nli or HFNLIProvider(model_name=config.model_name)
        self.config = config
        self.entailment_threshold = config.entailment_threshold
        self.contradiction_threshold = config.contradiction_threshold

    # -------------------------- Public workflow -----------------------------

    async def analyze_conversation(
        self,
        messages: List[Message],
        side: Stance,
        conversation_id: int,
        topic: str,
    ):
        stance = Stance(side.upper())  # "PRO" o "CON"

        state = self._get_state(conversation_id)
        _print(
            '[analyze] conv_id=',
            conversation_id,
            ' stance=',
            side,
            ' topic=',
            _trunc(topic),
        )

        mapped = self._map_history(messages)
        out = self.judge_last_two_messages(
            conversation=mapped, stance=stance, topic=topic
        )
        _print('[analyze] judge=', out)

        # ✅ IMPORTANTE: incrementar y quizá concluir ANTES de llamar al LLM
        if self._is_positive_judgment(out):
            state.positive_judgements += 1
            _print(
                '[analyze] +concession (counted) -> total:', state.positive_judgements
            )

        if state.maybe_conclude():
            state.match_concluded = True
            return self._build_verdict()

        _print('[analyze] calling llm.debate() with', len(messages), 'msgs')
        reply = await self.llm.debate(messages=messages)
        _print('[analyze] llm reply chars=', len(reply))
        state.assistant_turns += 1

        if state.maybe_conclude():
            state.match_concluded = True
            return self._build_verdict()

        return reply.strip()

    # -------------------------- Internal helpers ----------------------------

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

    def _get_state(self, conversation_id: int) -> DebateState:
        return self._state.setdefault(conversation_id, DebateState())

    @staticmethod
    def _drop_questions(text: str) -> str:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        sents = [s for s in sents if not IS_QUESTION_RX.search(s)]
        return ' '.join(sents) if sents else text

    @staticmethod
    def _agg_max(scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if 'agg_max' in scores:
            return scores['agg_max']
        p2h, h2p = scores['p_to_h'], scores['h_to_p']
        labels = set(p2h.keys()) | set(h2p.keys())
        return {lbl: max(p2h.get(lbl, 0.0), h2p.get(lbl, 0.0)) for lbl in labels}

    def _is_contradiction_symmetric(self, scores: Dict[str, Dict[str, float]]) -> bool:
        agg = self._agg_max(scores)
        contra = agg.get('contradiction', 0.0)
        ent = agg.get('entailment', 0.0)
        neu = agg.get('neutral', 0.0)
        ok = (
            (contra >= self.contradiction_threshold)
            and (contra >= ent)
            and (contra + EPS_CONTRA_VS_NEU >= neu)
        )
        _print('[contra] agg=', _round3(agg), '->', ok)
        return ok

    def _has_support_either_direction(
        self, scores: Dict[str, Dict[str, float]]
    ) -> Tuple[bool, str]:
        def ok(d: Dict[str, float]) -> bool:
            ent = d.get('entailment', 0.0)
            neu = d.get('neutral', 0.0)
            contra = d.get('contradiction', 0.0)
            return (
                (ent >= contra + MARGIN_EC)
                and (ent + EPS_ENT >= neu)
                and (ent >= MIN_ENT)
            )

        ph = scores['p_to_h']
        hp = scores['h_to_p']
        ph_ok, hp_ok = ok(ph), ok(hp)
        chosen = (
            'p→h'
            if ph_ok and (ph.get('entailment', 0.0) >= hp.get('entailment', 0.0))
            else ('h→p' if hp_ok else '')
        )
        _print(
            '[support] p→h',
            _round3(ph),
            'ok=',
            ph_ok,
            '| h→p',
            _round3(hp),
            'ok=',
            hp_ok,
            'chosen=',
            chosen,
        )
        return (ph_ok or hp_ok), chosen

    def _alignment_and_scores_topic_aware(
        self,
        bot_text: str,
        user_text: str,
        bot_stance: Stance,
        topic: str,
    ) -> Tuple[str, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        bot_clean = self._drop_questions(bot_text)

        # Assistant vs user (pairwise)
        pair_scores = self.nli.bidirectional_scores(bot_clean, user_text)

        # Thesis (topic + stance) vs user
        thesis = self._bot_thesis(topic, bot_stance)
        thesis_scores = self.nli.bidirectional_scores(thesis, user_text)

        # Alignment desde la tesis
        if self._is_contradiction_symmetric(thesis_scores):
            align = 'OPPOSITE'
        else:
            supported, _dir = self._has_support_either_direction(thesis_scores)
            align = 'SAME' if supported else 'UNKNOWN'

        _print('[align] thesis=', thesis)
        _print(
            '[align] pair   p→h=',
            _round3(pair_scores['p_to_h']),
            'h→p=',
            _round3(pair_scores['h_to_p']),
            'agg=',
            _round3(self._agg_max(pair_scores)),
        )
        _print(
            '[align] thesis p→h=',
            _round3(thesis_scores['p_to_h']),
            'h→p=',
            _round3(thesis_scores['h_to_p']),
            'agg=',
            _round3(self._agg_max(thesis_scores)),
        )
        _print('[align] decided:', align)

        return align, pair_scores, thesis_scores

    def _is_on_topic(self, thesis_scores: Dict[str, Dict[str, float]]) -> bool:
        ph = thesis_scores['p_to_h']
        hp = thesis_scores['h_to_p']

        def has_signal(d: Dict[str, float]) -> bool:
            return (
                max(d.get('entailment', 0.0), d.get('contradiction', 0.0))
                >= TOPIC_SIGNAL_MIN
            ) or (d.get('neutral', 1.0) <= TOPIC_NEU_MAX)

        on = has_signal(ph) or has_signal(hp)
        _print('[topic] on_topic=', on, 'ph=', _round3(ph), 'hp=', _round3(hp))
        return on

    def judge_last_two_messages(
        self, conversation: List[dict], stance: Stance, topic: str
    ) -> Optional[Dict[str, Any]]:
        if not conversation:
            return None

        user_idx = self._latest_idx(conversation, 'user')
        if user_idx is None:
            return None

        bot_idx = self._latest_valid_assistant_before(conversation, user_idx)
        if bot_idx is None:
            return None

        user_txt = conversation[user_idx]['content']
        bot_txt = conversation[bot_idx]['content']
        user_wc = self._word_count(user_txt)

        _print('[judge] user_idx=', user_idx, 'bot_idx=', bot_idx)
        _print(
            '[judge] user_preview=', _trunc(user_txt), '| bot_preview=', _trunc(bot_txt)
        )

        align, pair_scores, thesis_scores = self._alignment_and_scores_topic_aware(
            bot_txt, user_txt, stance, topic
        )
        on_topic = self._is_on_topic(thesis_scores)

        # 1) Contradicción contra la tesis (permite override por contradicción extra fuerte si usuario corto)
        thesis_is_contra = self._is_contradiction_symmetric(thesis_scores)
        thesis_contra_p = self._agg_max(thesis_scores).get('contradiction', 0.0)

        if thesis_is_contra and (
            user_wc >= self.config.min_user_words
            or thesis_contra_p >= self.config.strict_contra_threshold
        ):
            _print('[judge] DECISION: concession=True (thesis_opposition)')
            return {
                'passed_stance': stance.value,
                'alignment': 'OPPOSITE',
                'concession': True,
                'reason': 'thesis_opposition',
                'reasons': ['thesis_opposition']
                + (
                    ['pairwise_opposition']
                    if self._is_contradiction_symmetric(pair_scores)
                    else []
                ),
                'scores': self._agg_max(pair_scores),
                'thesis_scores': self._agg_max(thesis_scores),
                'user_text_sample': user_txt,
                'bot_text_sample': bot_txt,
                'topic': topic,
            }

        # 2) Soporte de la tesis (no cuenta)
        supported, _ = self._has_support_either_direction(thesis_scores)
        if supported:
            _print('[judge] DECISION: same_stance (no concession)')
            return {
                'passed_stance': stance.value,
                'alignment': 'SAME',
                'concession': False,
                'reason': 'same_stance',
                'reasons': ['same_stance'],
                'scores': self._agg_max(pair_scores),
                'thesis_scores': self._agg_max(thesis_scores),
                'user_text_sample': user_txt,
                'bot_text_sample': bot_txt,
                'topic': topic,
            }

        # 3) Fallback por pares (requiere on-topic y mínimo de palabras)
        if (
            on_topic
            and self._is_contradiction_symmetric(pair_scores)
            and user_wc >= self.config.min_user_words
        ):
            _print('[judge] DECISION: concession=True (pairwise_opposition)')
            return {
                'passed_stance': stance.value,
                'alignment': 'OPPOSITE',
                'concession': True,
                'reason': 'pairwise_opposition',
                'reasons': ['pairwise_opposition'],
                'scores': self._agg_max(pair_scores),
                'thesis_scores': self._agg_max(thesis_scores),
                'user_text_sample': user_txt,
                'bot_text_sample': bot_txt,
                'topic': topic,
            }

        # 4) Demasiado corto
        if user_wc < self.config.min_user_words:
            _print('[judge] DECISION: too_short')
            return {
                'passed_stance': stance.value,
                'alignment': 'UNKNOWN',
                'concession': False,
                'reason': 'too_short',
                'reasons': ['too_short'],
                'scores': self._agg_max(pair_scores),
                'thesis_scores': self._agg_max(thesis_scores),
                'user_text_sample': user_txt,
                'bot_text_sample': bot_txt,
                'topic': topic,
            }

        # 5) Off-topic explícito
        if not on_topic:
            _print('[judge] DECISION: off_topic')
            return {
                'passed_stance': stance.value,
                'alignment': 'UNKNOWN',
                'concession': False,
                'reason': 'off_topic',
                'reasons': ['off_topic'],
                'scores': self._agg_max(pair_scores),
                'thesis_scores': self._agg_max(thesis_scores),
                'user_text_sample': user_txt,
                'bot_text_sample': bot_txt,
                'topic': topic,
            }

        # 6) Indeterminado
        _print('[judge] DECISION: underdetermined')
        return {
            'passed_stance': stance.value,
            'alignment': 'UNKNOWN',
            'concession': False,
            'reason': 'underdetermined',
            'reasons': ['underdetermined'],
            'scores': self._agg_max(pair_scores),
            'thesis_scores': self._agg_max(thesis_scores),
            'user_text_sample': user_txt,
            'bot_text_sample': bot_txt,
            'topic': topic,
        }

    # ----------------------- Conversation navigation ------------------------

    def _latest_idx(
        self, conv: List[dict], role: str, *, before_idx: Optional[int] = None
    ) -> Optional[int]:
        for i in range(len(conv) - 1, -1, -1):
            if before_idx is not None and i >= before_idx:
                continue
            if conv[i].get('role') == role:
                return i
        return None

    def _latest_valid_assistant_before(
        self, conv: List[dict], before_idx: int, *, min_words: int = 10
    ) -> Optional[int]:
        for i in range(before_idx - 1, -1, -1):
            m = conv[i]
            if m.get('role') != 'assistant':
                continue
            words = [w for w in m.get('content', '').split() if w.isalpha()]
            if len(words) >= min_words:
                return i
        return None

    # --------------------------- Misc utilities -----------------------------

    @staticmethod
    def _is_positive_judgment(eval_payload: Optional[dict]) -> bool:
        return bool(eval_payload and eval_payload.get('concession'))

    @staticmethod
    def _bot_thesis(topic: str, bot_stance: Stance) -> str:
        t = topic.strip().rstrip('.')
        if bot_stance == Stance.PRO:
            return f'{t}.'
        else:
            return f'It is not true that {t}.'

    @staticmethod
    def _build_verdict() -> str:
        return (
            'On balance, the opposing argument addressed key counters with evidence and causality. '
            'I concede the point. Match concluded.'
        )

    def _word_count(self, text: str) -> int:
        return len(WORD_RX.findall(text))
