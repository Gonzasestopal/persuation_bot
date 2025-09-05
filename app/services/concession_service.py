import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.concession_policy import DebateState
from app.domain.models import Message
from app.domain.ports.llm import LLMPort
from app.verdicts import after_end_message, build_verdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # set to INFO en producción

# ----------------------------- Config & Enums -------------------------------


@dataclass(frozen=True)
class _NLIConfig:
    model_name: str = 'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli'
    entailment_threshold: float = 0.65
    contradiction_threshold: float = 0.70
    max_length: int = 512
    max_claims_per_turn: int = 3
    min_user_words: int = 8  # piso de palabras del usuario
    strict_contra_threshold: float = (
        0.90  # contradicción “extra fuerte” para textos cortos
    )
    margin_ec: float = 0.02  # margen E vs C
    eps_ent: float = 0.20  # ent debe superar neu por este margen
    eps_contra_vs_neu: float = 0.03  # contra no debe ser < neu por más que este eps
    min_ent_signal: float = 0.25  # señal mínima de entailment
    topic_neu_max: float = 0.75  # si neutral > esto, probablemente off-topic
    topic_signal_min: float = 0.30  # señal mínima (ent/contra) para on-topic


WORD_RX = re.compile(r'[^\W\d_]+', flags=re.UNICODE)  # palabras alfabéticas


class Stance(str, Enum):
    PRO = 'PRO'
    CON = 'CON'


END_MARKERS_RX = re.compile(
    r'(match concluded\.?|debate concluded|debate is over)', flags=re.IGNORECASE
)

SENT_SPLIT_RX = re.compile(r'(?<=[.!?¿\?¡!])\s+')
IS_QUESTION_RX = re.compile(r'[¿\?]\s*$')

# -------------------------------- Utilidades --------------------------------


def _trunc(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + '…'


def _round3(d: Dict[str, float]) -> Dict[str, float]:
    return {
        k: round(float(d.get(k, 0.0)), 3)
        for k in ('entailment', 'neutral', 'contradiction')
    }


def _normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def _sanitize_end_markers(text: str) -> str:
    return _normalize_spaces(END_MARKERS_RX.sub('', text))


# -------------------------------- Service -----------------------------------


class ConcessionService:
    """
    Usa NLI para decidir concesiones:
      - CONTRADICTION se decide simétricamente (agg de ambas direcciones).
      - SUPPORT (entailment) es direccional (mejor de dos), pero NO cuenta para concesión.
    """

    def __init__(
        self,
        llm: LLMPort,
        config: _NLIConfig = _NLIConfig(),
        nli: Optional[HFNLIProvider] = None,
        state: Optional[Dict[int, DebateState]] = None,
    ) -> None:
        self._state = state or {}
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

        state = self._get_state(conversation_id, stance)
        logger.debug(
            '[analyze] conv_id=%s stance=%s topic=%s',
            conversation_id,
            stance.value,
            _trunc(topic),
        )

        if state.match_concluded:
            logger.debug('[analyze] already ended → AFTER_END_MESSAGE')
            return after_end_message(state=state)

        mapped = self._map_history(messages)
        out = self.judge_last_two_messages(
            conversation=mapped, stance=stance, topic=topic
        )
        logger.debug('[analyze] judge=%s', out)

        # Incrementar señales ANTES del LLM
        if self._is_positive_judgment(out):
            state.positive_judgements += 1
            logger.debug('[analyze] +concession -> total=%d', state.positive_judgements)

        if state.maybe_conclude():
            state.match_concluded = True
            return build_verdict(state=state)

        logger.debug('[analyze] calling llm.debate() with %d msgs', len(messages))
        reply = await self.llm.debate(messages=messages, state=state)
        logger.debug('[analyze] llm reply chars=%d', len(reply))

        reply = _sanitize_end_markers(reply)

        state.assistant_turns += 1

        if state.maybe_conclude():
            state.match_concluded = True
            return build_verdict(state=state)

        return reply.strip()

    # -------------------------- Internal helpers ----------------------------

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

    def _get_state(self, conversation_id: int, stance: Stance) -> DebateState:
        # Ajusta si tu DebateState no tiene campo "stance"
        if conversation_id not in self._state:
            self._state[conversation_id] = DebateState(
                stance=stance.value
            )  # o sin stance si no existe
        return self._state[conversation_id]

    @staticmethod
    def _drop_questions(text: str) -> str:
        sents = [s.strip() for s in re.split(SENT_SPLIT_RX, text) if s.strip()]
        sents = [s for s in sents if not IS_QUESTION_RX.search(s)]
        out = ' '.join(sents) if sents else text
        # elimina dobles puntos finales accidentales
        out = re.sub(r'\.\.+$', '.', out).strip()
        return out

    @staticmethod
    def _agg_max(scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        # Asegura presencia de claves
        p2h = scores.get('p_to_h', {}) or {}
        h2p = scores.get('h_to_p', {}) or {}
        labels = {'entailment', 'neutral', 'contradiction'}
        return {
            lbl: max(float(p2h.get(lbl, 0.0)), float(h2p.get(lbl, 0.0)))
            for lbl in labels
        }

    def _is_contradiction_symmetric(self, scores: Dict[str, Dict[str, float]]) -> bool:
        agg = self._agg_max(scores)
        contra = agg.get('contradiction', 0.0)
        ent = agg.get('entailment', 0.0)
        neu = agg.get('neutral', 0.0)
        ok = (
            (contra >= self.contradiction_threshold)
            and (contra >= ent)
            and (contra + self.config.eps_contra_vs_neu >= neu)
        )
        logger.debug('[contra] agg=%s -> %s', _round3(agg), ok)
        return ok

    def _has_support_either_direction(
        self, scores: Dict[str, Dict[str, float]]
    ) -> Tuple[bool, str]:
        def ok(d: Dict[str, float]) -> bool:
            ent = float(d.get('entailment', 0.0))
            neu = float(d.get('neutral', 0.0))
            contra = float(d.get('contradiction', 0.0))
            # requisitos más exigentes para evitar SAME espurio
            return (
                ent >= max(contra + self.config.margin_ec, 0.70)  # ent >= 0.70
                and ent >= neu + max(self.config.eps_ent, 0.25)  # separar de neutral
                and contra <= 0.40  # contradicción baja
            )

        ph = scores.get('p_to_h', {}) or {}
        hp = scores.get('h_to_p', {}) or {}
        ph_ok, hp_ok = ok(ph), ok(hp)
        chosen = (
            'p→h'
            if ph_ok
            and (float(ph.get('entailment', 0.0)) >= float(hp.get('entailment', 0.0)))
            else ('h→p' if hp_ok else '')
        )
        logger.debug(
            '[support] p→h %s ok=%s | h→p %s ok=%s chosen=%s',
            _round3(ph),
            ph_ok,
            _round3(hp),
            hp_ok,
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
        user_clean = _normalize_spaces(user_text)

        # Assistant vs user (pairwise)
        pair_scores = self.nli.bidirectional_scores(bot_clean, user_clean)

        # Thesis (topic + stance) vs user
        thesis = self._bot_thesis(topic, bot_stance)
        thesis_scores = self.nli.bidirectional_scores(thesis, user_clean)

        # --- NUEVO: refuerzo por mejor oración (evita promedios engañosos) ---
        best_contra = self._max_contra_sentence(thesis, user_clean)
        if best_contra >= max(self.contradiction_threshold, 0.80):
            # Forzamos un paquete de scores que refleje contradicción fuerte
            thesis_scores = {
                'p_to_h': {
                    'entailment': 0.0,
                    'neutral': 0.0,
                    'contradiction': best_contra,
                },
                'h_to_p': {
                    'entailment': 0.0,
                    'neutral': 0.0,
                    'contradiction': best_contra,
                },
                'agg_max': {
                    'entailment': 0.0,
                    'neutral': 0.0,
                    'contradiction': best_contra,
                },
            }
        # ----------------------------------------------------------------------

        # Alignment desde la tesis (tu lógica original)
        if self._is_contradiction_symmetric(thesis_scores):
            align = 'OPPOSITE'
        else:
            supported, _dir = self._has_support_either_direction(thesis_scores)
            align = 'SAME' if supported else 'UNKNOWN'

        logger.debug('[align] thesis=%s', thesis)
        logger.debug(
            '[align] pair   p→h=%s h→p=%s agg=%s',
            _round3(pair_scores.get('p_to_h', {})),
            _round3(pair_scores.get('h_to_p', {})),
            _round3(self._agg_max(pair_scores)),
        )
        logger.debug(
            '[align] thesis p→h=%s h→p=%s agg=%s (best_contra=%.3f)',
            _round3(thesis_scores.get('p_to_h', {})),
            _round3(thesis_scores.get('h_to_p', {})),
            _round3(self._agg_max(thesis_scores)),
            best_contra,
        )
        logger.debug('[align] decided: %s', align)

        return align, pair_scores, thesis_scores

    def _is_on_topic(self, thesis_scores: Dict[str, Dict[str, float]]) -> bool:
        ph = thesis_scores.get('p_to_h', {}) or {}
        hp = thesis_scores.get('h_to_p', {}) or {}

        def has_signal(d: Dict[str, float]) -> bool:
            ent = float(d.get('entailment', 0.0))
            contra = float(d.get('contradiction', 0.0))
            neu = float(d.get('neutral', 1.0))
            return (max(ent, contra) >= self.config.topic_signal_min) or (
                neu <= self.config.topic_neu_max
            )

        on = has_signal(ph) or has_signal(hp)
        logger.debug('[topic] on_topic=%s ph=%s hp=%s', on, _round3(ph), _round3(hp))
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

        logger.debug('[judge] user_idx=%s bot_idx=%s', user_idx, bot_idx)
        logger.debug(
            '[judge] user_preview=%s | bot_preview=%s',
            _trunc(user_txt),
            _trunc(bot_txt),
        )

        align, pair_scores, thesis_scores = self._alignment_and_scores_topic_aware(
            bot_txt, user_txt, stance, topic
        )
        on_topic = self._is_on_topic(thesis_scores)

        # 1) Contradicción contra la tesis (permite override por contradicción extra fuerte si texto corto)
        thesis_is_contra = self._is_contradiction_symmetric(thesis_scores)
        thesis_contra_p = self._agg_max(thesis_scores).get('contradiction', 0.0)

        if thesis_is_contra and (
            user_wc >= self.config.min_user_words
            or thesis_contra_p >= self.config.strict_contra_threshold
        ):
            logger.debug('[judge] DECISION: concession=True (thesis_opposition)')
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

        # 2) Soporte de la tesis (no cuenta para concesión)
        supported, _ = self._has_support_either_direction(thesis_scores)
        if supported:
            logger.debug('[judge] DECISION: same_stance (no concession)')
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
            logger.debug('[judge] DECISION: concession=True (pairwise_opposition)')
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
            logger.debug('[judge] DECISION: too_short')
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
            logger.debug('[judge] DECISION: off_topic')
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
        logger.debug('[judge] DECISION: underdetermined')
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
            # usa WORD_RX (soporta español y evita números/guiones)
            words = WORD_RX.findall(m.get('content', ''))
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
        # Evita dejar ".." si ya venía con punto
        if bot_stance == Stance.PRO:
            return f'{t}.'
        else:
            return f'It is not true that {t}.'

    def _word_count(self, text: str) -> int:
        # cuenta solo tokens alfabéticos (multi-idioma)
        return len(WORD_RX.findall(text))

    def _max_contra_sentence(self, premise: str, hypothesis: str) -> float:
        """Devuelve la mayor prob. de contradicción tomando cada oración del hypothesis por separado."""
        sents = [s.strip() for s in re.split(SENT_SPLIT_RX, hypothesis) if s.strip()]
        best = 0.0
        for s in sents:
            sc = self.nli.bidirectional_scores(premise, s)
            agg = self._agg_max(sc)
            best = max(best, float(agg.get('contradiction', 0.0)))
        return best
