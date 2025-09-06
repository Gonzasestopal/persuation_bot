from app.domain.nli.types import NLIGradedSignal
from app.nli.ops import agg_max, is_contradiction_symmetric


def build_graded_signal(
    *,
    pairwise_scores: dict,
    similarity: float,
    on_topic: bool,
    user_wc: int,
    is_question_only: bool,
) -> NLIGradedSignal:
    con = float(pairwise_scores.get('contradiction', 0.0))
    ent = float(pairwise_scores.get('entailment', 0.0))
    score = con  # contradiction-first
    return NLIGradedSignal(
        score=score,
        similarity=float(similarity),
        on_topic=bool(on_topic),
        contradiction=con,
        entailment=ent,
        user_wc=int(user_wc),
        is_question_only=bool(is_question_only),
    )
