from dataclasses import dataclass
from enum import Enum


class ConcessionTier(str, Enum):
    NONE = 'NONE'  # continue normally
    SOFT = 'SOFT'  # acknowledge point, no state flip
    PARTIAL = 'PARTIAL'  # concede specific sub-claim
    FULL = 'FULL'  # concede the debate (end)


@dataclass(frozen=True)
class NLIGradedSignal:
    # core
    score: float
    similarity: float
    on_topic: bool
    # raw (optional but handy for logs)
    contradiction: float
    entailment: float
    # input-quality (for policy gates)
    user_wc: int
    is_question_only: bool
