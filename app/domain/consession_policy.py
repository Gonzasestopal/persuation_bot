from dataclasses import dataclass


@dataclass(frozen=True)
class ConcessionPolicy:
    min_assistant_turns_before_verdict: int
    required_positive_judgements: int


@dataclass
class DebateState:
    policy: ConcessionPolicy = ConcessionPolicy
    assistant_turns: int = 0
    positive_judgements: int = 0
    match_concluded: bool = False

    def _should_end(self):
        return (
            self.assistant_turns >= self.policy.min_assistant_turns_before_verdict and
            self.positive_judgements >= self.policy.required_positive_judgements
        )
