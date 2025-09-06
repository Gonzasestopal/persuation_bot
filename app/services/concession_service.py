# app/services/concession_service.py
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.domain.concession_policy import DebateState
from app.domain.enums import Stance
from app.domain.models import Message
from app.domain.nli.config import NLIConfig
from app.domain.nli.scoring import ScoringConfig
from app.domain.nli.types import ConcessionTier
from app.domain.ports.debate_store import DebateStorePort
from app.domain.ports.llm import LLMPort
from app.domain.ports.nli import NLIPort
from app.domain.verdicts import after_end_message, build_verdict
from app.nli.ops import agg_max
from app.services.concession_policy_engine import ConcessionPolicyConfig, apply_policy
from app.services.nli_graded import build_graded_signal
from app.utils.text import (
    drop_questions,
    normalize_spaces,
    round3,
    sanitize_end_markers,
    trunc,
    word_count,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ConcessionService:
    """
    Debate concession logic with graded NLI signals + policy tiers.

    Main flow used by MessageService.continue_conversation():
      - analyze_conversation() fetches DebateState and recent messages
      - Pre-process with analyze_turn(): builds graded NLI signal and policy tier
      - Steers LLM generation using (guidance, response_mode)
      - Optional post hooks / conclusion

    Back-compat:
      - judge_last_two_messages() retained (binary alignment), handy for diagnostics/tests
    """

    # Acknowledgment openers that should not be treated as claims
    ACK_PREFIXES = (
        "you're right",
        'you are right',
        'indeed',
        'i agree',
        'good point',
        'correct',
        'fair point',
        'exactly',
    )

    def __init__(
        self,
        llm: Optional[LLMPort] = None,
        nli: Optional[NLIPort] = None,
        nli_config: Optional[NLIConfig] = None,
        scoring: Optional[ScoringConfig] = None,
        debate_store: Optional[DebateStorePort] = None,
        policy_config: Optional[ConcessionPolicyConfig] = None,
    ) -> None:
        self.llm = llm
        self.nli = nli
        self.nli_config = nli_config or NLIConfig()
        self.scoring = scoring or ScoringConfig()
        self.debate_store = debate_store
        self.policy_config = policy_config or ConcessionPolicyConfig()
        # internal alias maintained for compatibility
        self.policy_cfg = self.policy_config

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def analyze_conversation(
        self, messages: List[Message], stance: Stance, conversation_id: int, topic: str
    ) -> str:
        """
        Main entry from MessageService.continue_conversation()
        """
        state = self.debate_store.get(conversation_id) if self.debate_store else None
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

        # Map repo messages -> {role, content}
        mapped: List[dict] = [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

        # Identify most recent user and a substantive earlier assistant message
        user_idx = next(
            (
                i
                for i in range(len(mapped) - 1, -1, -1)
                if mapped[i].get('role') == 'user'
            ),
            None,
        )
        bot_idx = None
        if user_idx is not None:
            bot_idx = next(
                (
                    i
                    for i in range(user_idx - 1, -1, -1)
                    if mapped[i].get('role') == 'assistant'
                    and word_count(mapped[i].get('content', '')) >= 10
                ),
                None,
            )

        # If we don't have a previous assistant message, just ask LLM to continue.
        if user_idx is None or bot_idx is None:
            logger.debug('[analyze] no prior assistant turn; skipping graded policy.')
            reply = await self._safe_llm_debate(messages=mapped)
            reply = sanitize_end_markers(reply)
            state.assistant_turns += 1
            self._maybe_finish_and_persist(conversation_id, state)
            return reply.strip()

        user_txt = mapped[user_idx]['content']
        bot_txt = mapped[bot_idx]['content']

        clean_topic = self._clean_topic_for_nli(topic)
        thesis = self._canonical_stance(clean_topic, stance)

        decision = await self.analyze_turn(
            state=state, user_msg=user_txt, bot_msg=bot_txt, thesis=thesis
        )

        logger.info(
            '[policy] cid=%s tier=%s pos_judg=%s signal={contra=%.3f ent=%.3f sim=%.3f topic=%s}',
            conversation_id,
            decision['tier'].name,
            state.positive_judgements,
            decision['signal']['contradiction'],
            decision['signal']['entailment'],
            decision['signal']['similarity'],
            decision['signal']['on_topic'],
        )

        tier: ConcessionTier = decision['tier']

        # Count concessions towards ending: PARTIAL/FULL only (SOFT doesn't)
        if tier in (ConcessionTier.PARTIAL, ConcessionTier.FULL):
            state.positive_judgements += 1

        guidance = self._guidance_from_tier(tier)
        response_mode = {
            ConcessionTier.NONE: 'defend',
            ConcessionTier.SOFT: 'soft_concede',
            ConcessionTier.PARTIAL: 'partial_concede',
            ConcessionTier.FULL: 'full_concede',
        }[tier]

        logger.debug(
            "[steer] cid=%s mode=%s guidance='%s'",
            conversation_id,
            response_mode,
            trunc(guidance, 140),
        )

        # ---- GEN: steer the model ----
        reply = await self.llm.debate(
            messages=mapped,
            guidance=guidance,
            response_mode=response_mode,
            state=state,
        )
        reply = sanitize_end_markers(reply)

        # If FULL, conclude immediately
        if tier == ConcessionTier.FULL:
            logger.info('[end] cid=%s full_concession -> concluding', conversation_id)
            state.match_concluded = True
            self.debate_store.save(conversation_id=conversation_id, state=state)
            return build_verdict(state=state)

        # ---- POST: bookkeeping and possible conclusion via state policy ----
        state.assistant_turns += 1
        self._maybe_finish_and_persist(conversation_id, state)
        return reply.strip()

    async def analyze_turn(
        self, state: DebateState, user_msg: str, bot_msg: str, thesis: str
    ) -> Dict[str, Any]:
        """
        One-turn graded analysis:
          - extract best claim vs user
          - NLI probs + similarity + topic gate
          - apply policy (EMA/streaks) -> tier
        """
        # 1) pick best claim pair (thesis fallback) and compute NLI + similarity + topic gate
        best_pair = await self._extract_best_claim_pair(
            user_msg, bot_msg, thesis
        )  # (premise, hypothesis)
        pairwise = await self._nli_probs(best_pair)
        similarity = await self._similarity(best_pair)  # [0,1]
        on_topic = await self._topic_gate(user_msg, thesis)  # bool

        logger.debug(
            "[nli] pair_premise='%s' pair_hyp='%s' probs={contra=%.3f ent=%.3f neu=%.3f} sim=%.3f topic=%s",
            trunc(best_pair[0], 120),
            trunc(best_pair[1], 120),
            pairwise.get('contradiction', 0.0),
            pairwise.get('entailment', 0.0),
            pairwise.get('neutral', 0.0),
            similarity,
            on_topic,
        )

        # 2) graded signal (contradiction-first) + input-quality features
        u_wc = word_count(user_msg)
        is_q_only = user_msg.strip().endswith('?') and u_wc <= getattr(
            self.policy_cfg, 'question_only_wc_max', 6
        )

        signal = build_graded_signal(
            pairwise_scores=pairwise,
            similarity=similarity,
            on_topic=on_topic,
            user_wc=u_wc,
            is_question_only=is_q_only,
        )

        logger.debug(
            '[signal] contra=%.3f ent=%.3f score=%.3f sim=%.3f on_topic=%s',
            pairwise.get('contradiction', 0.0),
            pairwise.get('entailment', 0.0),
            signal.score,
            signal.similarity,
            on_topic,
        )

        # 3) policy decision (updates state with EMA/streaks internally)
        tier = apply_policy(state=state, signal=signal, cfg=self.policy_cfg)

        # 4) structured response for telemetry / UI (optional)
        rationale = {
            ConcessionTier.FULL: 'High, sustained contradiction on-topic. Ending debate.',
            ConcessionTier.PARTIAL: 'Sustained strong contradiction on a sub-claim.',
            ConcessionTier.SOFT: 'Notable contradiction but below sustained threshold.',
            ConcessionTier.NONE: 'No actionable contradiction.',
        }[tier]

        return {
            'tier': tier,
            'rationale': rationale,
            'signal': {
                'contradiction': round(pairwise.get('contradiction', 0.0), 3),
                'entailment': round(pairwise.get('entailment', 0.0), 3),
                'similarity': round(similarity, 3),
                'on_topic': on_topic,
            },
        }

    # ---------------------------------------------------------------------
    # Legacy binary judge (kept for diagnostics / tests)
    # ---------------------------------------------------------------------

    def judge_last_two_messages(
        self, conversation: List[dict], stance: Stance, topic: str
    ) -> Optional[Dict[str, Any]]:
        """
        Older binary judge that returns {'alignment': 'OPPOSITE'|'SAME'|'UNKNOWN', 'concession': bool, ...}
        Useful for diagnostics; main path now uses analyze_turn() + policy tiers.
        """
        if not conversation:
            return None

        # last user
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

        # most recent substantive assistant before that (>=10 words)
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
        user_clean = normalize_spaces(user_txt)

        # Extract claims from assistant reply (used for pairwise)
        claims = self._extract_claims(bot_txt)

        # Canonical thesis sentences for stance (self/opp)
        clean_topic = self._clean_topic_for_nli(topic)
        canon_self = self._canonical_stance(clean_topic, stance)
        canon_opp = self._canonical_stance(
            clean_topic, Stance.CON if stance == Stance.PRO else Stance.PRO
        )

        # --- 1) Pairwise (claims vs user), pick strongest contradiction
        if claims:
            claim_scores = self._claim_scores(claims, user_clean)
            best_by_contra = max(claim_scores, key=lambda t: t[2])
            best_claim, best_entail, best_contra, best_related, best_pair_scores = (
                best_by_contra
            )
        else:
            best_claim = ''
            best_entail = 0.0
            best_contra = 0.0
            best_related = 0.0
            best_pair_scores = {'p_to_h': {}, 'h_to_p': {}}

        relatedness_min = getattr(self.scoring, 'relatedness_min', 0.35)
        engaged = best_related >= relatedness_min
        logger.debug(
            "[claims] n=%d | best_contra=%.3f '%s'",
            len(claims),
            best_contra,
            trunc(best_claim, 60),
        )

        # --- 2) Thesis-level scores (canonical self vs user)
        self_scores = (
            self.nli.bidirectional_scores(canon_self, user_clean) if self.nli else {}
        )
        self_agg = agg_max(self_scores) if self.nli else {}

        # Topic gate: from thesis scores
        on_topic = self._on_topic_from_scores(self_scores) if self.nli else True

        # Sentence-level scan: max contradiction across user sentences
        max_sent_contra, max_sent_ent, max_sent_scores = (
            self._max_contra_self_vs_sentences(canon_self, user_txt)
            if self.nli
            else (0.0, 0.0, {})
        )

        # --- 3) Opposite stance support checks
        opp_scores = (
            self.nli.bidirectional_scores(canon_opp, user_clean) if self.nli else {}
        )
        opp_agg = agg_max(opp_scores) if self.nli else {}

        support_min = getattr(self.scoring, 'support_min', 0.50)
        contra_min = getattr(self.scoring, 'strict_contra_threshold', 0.55)
        pair_soft = getattr(self.scoring, 'contradiction_threshold', 0.55)
        margin_min = getattr(self.scoring, 'margin_min', 0.15)
        min_user_words = getattr(self.scoring, 'min_user_words', 8)

        self_ent = float(self_agg.get('entailment', 0.0) if self_agg else 0.0)
        self_con = float(self_agg.get('contradiction', 0.0) if self_agg else 0.0)
        opp_ent = float(opp_agg.get('entailment', 0.0) if opp_agg else 0.0)
        opp_con = float(opp_agg.get('contradiction', 0.0) if opp_agg else 0.0)

        # User supports opposite if: entail opposite OR contradict self (with margin)
        opp_supported = (
            opp_ent >= support_min and (opp_ent - opp_con) >= margin_min
        ) or (self_con >= contra_min and (self_con - self_ent) >= margin_min)

        # User supports our stance?
        self_supported = (self_ent >= support_min) and (
            (self_ent - self_con) >= margin_min
        )

        logger.debug(
            '[topic] on_topic=%s | agg=%s',
            on_topic,
            round3(self_agg),
        )
        logger.debug(
            "[rel] best_claim_relatedness=%.3f (min=%.3f) | best_claim='%s'",
            best_related,
            relatedness_min,
            trunc(best_claim, 60),
        )

        # ------------------- Decision branches (ordered) -------------------

        # A) Strong thesis contradiction → concede (overrides min words)
        thesis_strong_contra = max(self_con, max_sent_contra) >= contra_min
        if thesis_strong_contra and on_topic:
            return self._mk_result(
                stance=stance,
                alignment='OPPOSITE',
                concession=True,
                reason='thesis_opposition_soft',
                pair_scores=max_sent_scores or self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # B) SAME if thesis supported (non-conceding)
        if on_topic and self_supported:
            return self._mk_result(
                stance=stance,
                alignment='SAME',
                concession=False,
                reason='thesis_support',
                pair_scores=self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # C) Too short blocks other concessions (except A)
        if user_wc < min_user_words:
            return self._mk_result(
                stance=stance,
                alignment='UNKNOWN',
                concession=False,
                reason='too_short',
                pair_scores=best_pair_scores or self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # D) Off-topic blocks pairwise fallback
        if not on_topic:
            return self._mk_result(
                stance=stance,
                alignment='UNKNOWN',
                concession=False,
                reason='off_topic',
                pair_scores=best_pair_scores or self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # E) Pairwise contradiction fallback (soft) → concede
        if engaged and best_contra >= pair_soft:
            return self._mk_result(
                stance=stance,
                alignment='OPPOSITE',
                concession=True,
                reason='pairwise_opposition_soft',
                pair_scores=best_pair_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # F) Otherwise unknown / underdetermined
        return self._mk_result(
            stance=stance,
            alignment='UNKNOWN',
            concession=False,
            reason='underdetermined',
            pair_scores=best_pair_scores or self_scores,
            thesis_scores=self_scores,
            user_txt=user_txt,
            bot_txt=bot_txt,
            topic=clean_topic,
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    async def _extract_best_claim_pair(
        self, user_msg: str, bot_msg: str, thesis: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Build candidate pairs (claim→user, plus thesis→user fallback),
        score them via NLI, and pick by highest contradiction with an
        engagement floor. If NLI is unavailable, pick first candidate or fallback.
        """
        user_clean = normalize_spaces(user_msg)
        claims = self._extract_claims(bot_msg)

        candidates: List[Tuple[str, str]] = [(c, user_clean) for c in claims]
        if thesis:
            candidates.append((thesis, user_clean))  # robust fallback

        # If no NLI, pick a safe fallback
        if not self.nli:
            if candidates:
                return candidates[0]
            # last fallback: avoid (user,user) if possible
            return (thesis or user_clean, user_clean)

        # Score candidates; choose by highest contradiction, tiebreak by relatedness
        best: Optional[Tuple[str, str]] = None
        best_con, best_rel = -1.0, -1.0

        for p, h in candidates:
            sc = self.nli.bidirectional_scores(p, h)
            agg = agg_max(sc)
            con = float(agg.get('contradiction', 0.0))
            ent = float(agg.get('entailment', 0.0))
            neu = float(agg.get('neutral', 1.0))
            rel = max(ent, con, 1.0 - neu)  # similarity proxy
            if (con, rel) > (best_con, best_rel):
                best, best_con, best_rel = (p, h), con, rel

        # Engagement floor: if weak, force thesis→user
        sim_min = getattr(self.policy_cfg, 'similarity_min', 0.60)
        if best and best_rel >= sim_min:
            return best
        if thesis:
            return (thesis, user_clean)
        return (user_clean, user_clean)

    async def _nli_probs(self, pair: Tuple[str, str]) -> Dict[str, float]:
        if not self.nli:
            return {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0}
        p, h = pair
        sc = self.nli.bidirectional_scores(p, h)
        agg = agg_max(sc)
        return {
            'entailment': float(agg.get('entailment', 0.0)),
            'contradiction': float(agg.get('contradiction', 0.0)),
            'neutral': float(agg.get('neutral', 0.0)),
        }

    async def _similarity(self, pair: Tuple[str, str]) -> float:
        if not self.nli:
            return 0.0
        p, h = pair
        sc = self.nli.bidirectional_scores(p, h)
        agg = agg_max(sc)
        ent = float(agg.get('entailment', 0.0))
        con = float(agg.get('contradiction', 0.0))
        neu = float(agg.get('neutral', 1.0))
        return max(ent, con, 1.0 - neu)

    async def _topic_gate(self, user_msg: str, thesis: str) -> bool:
        if not self.nli:
            return True
        user_clean = normalize_spaces(user_msg)
        thesis_scores = self.nli.bidirectional_scores(thesis, user_clean)
        return self._on_topic_from_scores(thesis_scores)

    def _extract_claims(self, bot_txt: str) -> List[str]:
        """
        Bot format (guaranteed by caller):
          [0] stance header (drop)
          [1..-2] declarative claims (keep)
          [-1] trailing question (drop)
        """
        if not bot_txt:
            return []
        parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', bot_txt) if p.strip()]
        if not parts:
            return []

        # Drop first (stance header) and last (question)
        if len(parts) >= 2:
            parts = parts[1:-1]
        else:
            parts = []

        claims: List[str] = []
        for s in parts:
            s2 = drop_questions(s).strip()
            if not s2:
                continue
            s2_l = s2.lower()
            if any(s2_l.startswith(prefix) for prefix in self.ACK_PREFIXES):
                continue
            if not s2.endswith(('.', '!')):
                s2 += '.'
            if len(s2.split()) >= 3:
                claims.append(s2)
        logger.debug(
            "[claims] extracted=%d first='%s'",
            len(claims),
            trunc(claims[0] if claims else '', 120),
        )
        return claims

    def _claim_scores(
        self, claims: List[str], user_clean: str
    ) -> List[Tuple[str, float, float, float, Dict[str, Dict[str, float]]]]:
        out: List[Tuple[str, float, float, float, Dict[str, Dict[str, float]]]] = []
        for c in claims:
            if not self.nli:
                out.append((c, 0.0, 0.0, 0.0, {}))
                continue
            sc = self.nli.bidirectional_scores(c, user_clean)
            agg = agg_max(sc)
            ent = float(agg.get('entailment', 0.0))
            con = float(agg.get('contradiction', 0.0))
            neu = float(agg.get('neutral', 1.0))
            rel = max(ent, con, 1.0 - neu)
            out.append((c, ent, con, rel, sc))
        return out

    def _on_topic_from_scores(self, thesis_scores: Dict[str, Dict[str, float]]) -> bool:
        ph = thesis_scores.get('p_to_h', {}) or {}
        hp = thesis_scores.get('h_to_p', {}) or {}

        def has_signal(d: Dict[str, float]) -> bool:
            ent = float(d.get('entailment', 0.0))
            con = float(d.get('contradiction', 0.0))
            neu = float(d.get('neutral', 1.0))
            return (max(ent, con) >= self.scoring.topic_signal_min) or (
                neu <= self.scoring.topic_neu_max
            )

        on = has_signal(ph) or has_signal(hp)
        logger.debug('[topic] on_topic=%s | agg=%s', on, round3(agg_max(thesis_scores)))
        return on

    @staticmethod
    def _polarity_variants(t: str) -> Tuple[str, str]:
        t0 = (t or '').strip().rstrip('.')
        tl = t0.lower()

        # exists ↔ does not exist
        m = re.match(r'^(.+?)\s+do(?:es)?\s+not\s+exist$', tl, flags=re.I)
        if m:
            subj = t0[: t0.lower().rfind(' does not exist')].strip()
            return f'{subj} exists.', f'{subj} does not exist.'
        m = re.match(r"^(.+?)\s+doesn'?t\s+exist$", tl, flags=re.I)
        if m:
            subj = t0[: t0.lower().rfind(" doesn't exist")].strip()
            return f'{subj} exists.', f'{subj} does not exist.'
        m = re.match(r'^(.+?)\s+exists$', tl, flags=re.I)
        if m:
            subj = t0[: t0.lower().rfind(' exists')].strip()
            return f'{subj} exists.', f'{subj} does not exist.'

        # are not / are
        m = re.match(r'^(?P<subj>.+?)\s+are\s+not\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} are {pred}.', f'{subj} are not {pred}.'
        m = re.match(r'^(?P<subj>.+?)\s+are\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} are {pred}.', f'{subj} are not {pred}.'

        # is not / is
        m = re.match(r'^(?P<subj>.+?)\s+is\s+not\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} is {pred}.', f'{subj} is not {pred}.'
        m = re.match(r'^(?P<subj>.+?)\s+is\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} is {pred}.', f'{subj} is not {pred}.'

        # fallback
        pos = t0 + '.'
        neg = f'It is not the case that {t0}.'
        return pos, neg

    @staticmethod
    def _canonical_stance(topic_clean: str, bot_stance: Stance) -> str:
        pos, neg = ConcessionService._polarity_variants(topic_clean)
        return pos if bot_stance == Stance.PRO else neg

    @staticmethod
    def _mk_result(
        stance: Stance,
        alignment: str,
        concession: bool,
        reason: str,
        pair_scores: Dict[str, Dict[str, float]],
        thesis_scores: Dict[str, Dict[str, float]],
        user_txt: str,
        bot_txt: str,
        topic: str,
    ) -> Dict[str, Any]:
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
    def _clean_topic_for_nli(topic: str) -> str:
        """
        Remove accidental meta like 'Language: EN', 'Side: PRO', 'Topic: ...'
        Keep only the proposition's first sentence.
        """
        s = topic
        s = re.sub(r'\b(Language|Idioma)\s*:\s*[A-Za-z]{2}\b\.?', '', s, flags=re.I)
        s = re.sub(r'\b(Side|Lado)\s*:\s*(PRO|CON)\b\.?', '', s, flags=re.I)
        s = re.sub(r'\b(Topic|Tema)\s*:\s*', '', s, flags=re.I)
        s = s.strip().strip('.')
        s = s.split('.')[0].strip()
        return s

    # ----------------------------- sentence scan -----------------------------

    def _max_contra_self_vs_sentences(
        self, canonical_self: str, user_txt: str
    ) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
        """
        Scan user sentences vs canonical self-thesis; return max contradiction and its scores.
        """
        if not self.nli:
            return 0.0, 0.0, {}

        sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', user_txt) if s.strip()
        ]
        best_contra = 0.0
        best_ent = 0.0
        best_scores: Dict[str, Dict[str, float]] = {}
        for s in sentences:
            sc = self.nli.bidirectional_scores(canonical_self, s)
            agg = agg_max(sc)
            con = float(agg.get('contradiction', 0.0))
            ent = float(agg.get('entailment', 0.0))
            if con > best_contra:
                best_contra = con
                best_ent = ent
                best_scores = sc
        return best_contra, best_ent, best_scores

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _guidance_from_tier(self, t: ConcessionTier) -> str:
        if t == ConcessionTier.SOFT:
            return (
                'Acknowledge a valid point briefly, then reinforce your core argument '
                'without changing your stance.'
            )
        if t == ConcessionTier.PARTIAL:
            return (
                'Concede a specific sub-point clearly, delimit the scope of that concession, '
                'and reframe your core thesis.'
            )
        if t == ConcessionTier.FULL:
            return 'Concede succinctly and end the debate politely.'
        return 'Defend the thesis with evidence and address the strongest objection directly.'

    async def _safe_llm_debate(
        self,
        *,
        messages: List[dict],
        guidance: Optional[str] = None,
        response_mode: Optional[str] = None,
        state: Optional[DebateState] = None,
    ) -> str:
        """
        Calls LLMPort.debate with best-effort compatibility:
          1) messages + guidance + response_mode + state
          2) messages + guidance + response_mode
          3) messages + guidance
          4) messages
        """
        if not self.llm:
            raise RuntimeError('LLMPort not configured')

        # 1) most expressive signature
        try:
            return await self.llm.debate(
                messages=messages,
                guidance=guidance,
                response_mode=response_mode,
                state=state,
            )
        except TypeError:
            pass

        # 2) without state
        try:
            return await self.llm.debate(
                messages=messages, guidance=guidance, response_mode=response_mode
            )
        except TypeError:
            pass

        # 3) only guidance
        try:
            return await self.llm.debate(messages=messages, guidance=guidance)
        except TypeError:
            pass

        # 4) minimal
        return await self.llm.debate(messages=messages)

    def _maybe_finish_and_persist(
        self, conversation_id: int, state: DebateState
    ) -> None:
        """
        Persist state and optionally conclude if state has an internal policy.
        """
        logger.debug(
            '[state] cid=%s turns=%s pos_judg=%s concluded=%s',
            conversation_id,
            state.assistant_turns,
            state.positive_judgements,
            state.match_concluded,
        )
        if getattr(state, 'maybe_conclude', None):
            try:
                if state.maybe_conclude():  # type: ignore[attr-defined]
                    state.match_concluded = True
            except Exception:
                logger.exception('maybe_conclude() raised; ignoring')
        if self.debate_store:
            self.debate_store.save(conversation_id=conversation_id, state=state)
