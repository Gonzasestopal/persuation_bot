# app/services/concession_policy_engine.py
from app.domain.concession_policy import ConcessionPolicyConfig
from app.domain.nli.types import ConcessionTier
from app.services.nli_graded import NLIGradedSignal


def _ema(prev, x, a):
    return x if prev is None else (1 - a) * prev + a * x


def apply_policy(
    *, state, signal: NLIGradedSignal, cfg: ConcessionPolicyConfig
) -> ConcessionTier:
    # -------- Input-quality gates (short / question-only) --------
    if signal.user_wc < cfg.min_user_words or (
        signal.is_question_only and signal.user_wc <= cfg.question_only_wc_max
    ):
        # Warm EMAs a touch so next turn isn't jumpy
        state.ema_contradiction = _ema(
            getattr(state, 'ema_contradiction', None), 0.0, cfg.ema_alpha
        )
        state.ema_similarity = _ema(
            getattr(state, 'ema_similarity', None), signal.similarity, cfg.ema_alpha
        )
        state.contradiction_streak_partial = 0
        state.contradiction_streak_full = 0
        return ConcessionTier.NONE

    # -------- Turn gate --------
    if getattr(state, 'assistant_turns', 0) < cfg.min_turns_before_any_concession:
        state.ema_contradiction = _ema(
            getattr(state, 'ema_contradiction', None), signal.score, cfg.ema_alpha
        )
        state.ema_similarity = _ema(
            getattr(state, 'ema_similarity', None), signal.similarity, cfg.ema_alpha
        )
        state.contradiction_streak_partial = 0
        state.contradiction_streak_full = 0
        return ConcessionTier.NONE

    # -------- Topic/similarity gates --------
    if cfg.require_on_topic and not signal.on_topic:
        state.ema_contradiction = _ema(
            getattr(state, 'ema_contradiction', None), 0.0, cfg.ema_alpha
        )
        state.ema_similarity = _ema(
            getattr(state, 'ema_similarity', None), signal.similarity, cfg.ema_alpha
        )
        state.contradiction_streak_partial = 0
        state.contradiction_streak_full = 0
        return ConcessionTier.NONE

    if signal.similarity < cfg.similarity_min:
        state.ema_contradiction = _ema(
            getattr(state, 'ema_contradiction', None), 0.0, cfg.ema_alpha
        )
        state.ema_similarity = _ema(
            getattr(state, 'ema_similarity', None), signal.similarity, cfg.ema_alpha
        )
        state.contradiction_streak_partial = 0
        state.contradiction_streak_full = 0
        return ConcessionTier.NONE

    # -------- Update EMAs --------
    state.ema_contradiction = _ema(
        getattr(state, 'ema_contradiction', None), signal.score, cfg.ema_alpha
    )
    state.ema_similarity = _ema(
        getattr(state, 'ema_similarity', None), signal.similarity, cfg.ema_alpha
    )

    # Ensure counters exist
    state.contradiction_streak_partial = getattr(
        state, 'contradiction_streak_partial', 0
    )
    state.contradiction_streak_full = getattr(state, 'contradiction_streak_full', 0)

    # -------- One-shot checks --------
    if signal.score >= cfg.full_contra_min:
        state.contradiction_streak_full += 1
        state.contradiction_streak_partial += 1
    elif signal.score >= cfg.partial_contra_min:
        state.contradiction_streak_partial += 1
        state.contradiction_streak_full = 0
        return (
            ConcessionTier.PARTIAL if cfg.partial_streak == 1 else ConcessionTier.SOFT
        )
    elif signal.score >= cfg.soft_contra_min:
        state.contradiction_streak_partial = 0
        state.contradiction_streak_full = 0
        return ConcessionTier.SOFT
    else:
        state.contradiction_streak_partial = 0
        state.contradiction_streak_full = 0

    # -------- Streak escalation --------
    if state.contradiction_streak_full >= cfg.full_streak:
        return ConcessionTier.FULL
    if state.contradiction_streak_partial >= cfg.partial_streak:
        return ConcessionTier.PARTIAL

    # -------- EMA backstops --------
    if state.ema_contradiction >= cfg.ema_full_min:
        return ConcessionTier.FULL
    if state.ema_contradiction >= cfg.ema_partial_min:
        return ConcessionTier.PARTIAL
    if state.ema_contradiction >= cfg.ema_soft_min:
        return ConcessionTier.SOFT

    return ConcessionTier.NONE
