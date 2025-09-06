from dataclasses import dataclass, field
from typing import List, Optional

from app.domain.enums import Stance
from app.domain.nli.types import ConcessionTier


@dataclass(frozen=True)
class ConcessionPolicy:
    # KO: end immediately on a FULL concession
    end_on_full: bool = True

    # Margin (recent quality)
    recent_window: int = 3  # look back this many tiers
    recent_min_positives: int = 2  # how many PARTIAL/FULL in that window
    ema_contra_min: float = 0.80  # require sustained pressure

    # Points (total quality)
    total_min_positives: int = 3  # cumulative PARTIAL/FULL needed
    require_recent_positive: bool = True  # don't end right after a NONE tier


@dataclass
class DebateState:
    stance: Stance
    lang: str
    topic: str
    policy: ConcessionPolicy = field(default_factory=ConcessionPolicy)

    # rolling outcome memory
    last_tier: Optional[ConcessionTier] = None
    last_k_tiers: List[ConcessionTier] = field(default_factory=list)

    assistant_turns: int = 0
    positive_judgements: int = 0
    match_concluded: bool = False
    lang_locked: bool = False  # once True, never auto-change

    # smoothing + streak counters
    ema_contradiction: Optional[float] = None
    ema_similarity: Optional[float] = None
    contradiction_streak_partial: int = 0
    contradiction_streak_full: int = 0

    # optional bookkeeping
    soft_concessions: int = 0
    partial_concessions: int = 0

    # helper: push most recent tier into ring buffer
    def push_tier(self, tier: ConcessionTier, max_keep: int = 5) -> None:
        self.last_tier = tier
        self.last_k_tiers.append(tier)
        # keep a small ring buffer (you can tie this to policy.recent_window if you like)
        if len(self.last_k_tiers) > max_keep:
            del self.last_k_tiers[: len(self.last_k_tiers) - max_keep]

    def should_end(self) -> bool:
        p = self.policy

        # KO lane
        if p.end_on_full and self.last_tier == ConcessionTier.FULL:
            return True

        # Recent window lane
        # (look back p.recent_window from the end of last_k_tiers)
        if p.recent_window > 0 and self.last_k_tiers:
            recent = self.last_k_tiers[-p.recent_window :]
            recent_pos = sum(
                t in (ConcessionTier.PARTIAL, ConcessionTier.FULL) for t in recent
            )
            if (
                recent_pos >= p.recent_min_positives
                and (self.ema_contradiction or 0.0) >= p.ema_contra_min
            ):
                return True

        # Points lane (cumulative)
        if self.positive_judgements >= p.total_min_positives:
            if (not p.require_recent_positive) or (
                self.last_tier in (ConcessionTier.PARTIAL, ConcessionTier.FULL)
            ):
                return True

        return False

    def maybe_conclude(self) -> bool:
        """Mark concluded if policy is satisfied; return current concluded flag."""
        if not self.match_concluded and self.should_end():
            self.match_concluded = True
        return self.match_concluded


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
    full_streak: int = 2
