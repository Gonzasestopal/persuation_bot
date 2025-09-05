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
    policy: ConcessionPolicy = field(default_factory=ConcessionPolicy)
    assistant_turns: int = 0
    positive_judgements: int = 0
    match_concluded: bool = False
    lang: str = 'en'  # fixed language for this conversation
    lang_locked: bool = False  # once True, never auto-change

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
