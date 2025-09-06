from dataclasses import dataclass, field

from app.domain.enums import Stance
from app.settings import settings


@dataclass(frozen=True)
class ConcessionPolicy:
    min_assistant_turns_before_verdict: int = (
        settings.MIN_ASSISTANT_TURNS_BEFORE_VERDICT
    )
    required_positive_judgements: int = settings.REQUIRED_POSITIVE_JUDGEMENTS


@dataclass
class DebateState:
    stance: Stance
    lang: str
    topic: str
    policy: ConcessionPolicy = field(default_factory=ConcessionPolicy)

    assistant_turns: int = 0
    positive_judgements: int = 0
    match_concluded: bool = False
    lang_locked: bool = False  # once True, never auto-change

    # NEW: smoothing + streak counters
    ema_contradiction: float = None
    ema_similarity: float = None
    contradiction_streak_partial: int = 0
    contradiction_streak_full: int = 0

    # optional bookkeeping
    soft_concessions: int = 0
    partial_concessions: int = 0

    def should_end(self) -> bool:
        return (
            self.assistant_turns >= self.policy.min_assistant_turns_before_verdict
            and self.positive_judgements >= self.policy.required_positive_judgements
        )

    def maybe_conclude(self) -> bool:
        """Mark concluded if policy is satisfied; return current concluded flag."""
        if not self.match_concluded and self.should_end():
            self.match_concluded = True
        return self.match_concluded


from dataclasses import dataclass


@dataclass
class ConcessionPolicyConfig:
    # engagement / input-quality gates
    min_user_words: int = 5
    question_only_wc_max: int = 6

    # topic/turn gates
    min_turns_before_any_concession: int = 0
    require_on_topic: bool = True
    similarity_min: float = 0.60

    # thresholds (contradiction-first)
    soft_contra_min: float = 0.60
    partial_contra_min: float = 0.75
    full_contra_min: float = 0.90

    # EMA + streaks
    ema_alpha: float = 0.5
    ema_soft_min: float = 0.65
    ema_partial_min: float = 0.78
    ema_full_min: float = 0.88
    partial_streak: int = 1
    full_streak: int = 2
