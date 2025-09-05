# app/nli/ops.py
import re
from typing import Dict, Tuple

from app.domain.nli.scoring import ScoringConfig
from app.utils.text import SENT_SPLIT_RX, round3


def agg_max(scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    p2h = scores.get('p_to_h', {}) or {}
    h2p = scores.get('h_to_p', {}) or {}
    labels = ('entailment', 'neutral', 'contradiction')
    return {
        lbl: max(float(p2h.get(lbl, 0.0)), float(h2p.get(lbl, 0.0))) for lbl in labels
    }


def is_contradiction_symmetric(
    scores: Dict[str, Dict[str, float]], cfg: ScoringConfig, *, logger=None
) -> bool:
    agg = agg_max(scores)
    contra, ent, neu = (
        agg.get('contradiction', 0.0),
        agg.get('entailment', 0.0),
        agg.get('neutral', 0.0),
    )
    ok = (
        (contra >= cfg.contradiction_threshold)
        and (contra >= ent)
        and (contra + cfg.eps_contra_vs_neu >= neu)
    )
    if logger:
        logger.debug('[contra] agg=%s -> %s', round3(agg), ok)
    return ok


def has_support_either_direction(
    scores: Dict[str, Dict[str, float]], cfg: ScoringConfig, *, logger=None
) -> Tuple[bool, str]:
    def ok(d: Dict[str, float]) -> bool:
        ent = float(d.get('entailment', 0.0))
        neu = float(d.get('neutral', 0.0))
        contra = float(d.get('contradiction', 0.0))
        return (
            ent >= max(contra + cfg.margin_ec, cfg.min_ent_for_same)
            and ent >= neu + max(cfg.eps_ent, cfg.margin_ent_vs_neu)
            and contra <= cfg.max_contra_for_same
        )

    ph = scores.get('p_to_h', {}) or {}
    hp = scores.get('h_to_p', {}) or {}
    ph_ok, hp_ok = ok(ph), ok(hp)
    chosen = (
        'p→h'
        if ph_ok
        and float(ph.get('entailment', 0.0)) >= float(hp.get('entailment', 0.0))
        else ('h→p' if hp_ok else '')
    )
    if logger:
        logger.debug(
            '[support] p→h %s ok=%s | h→p %s ok=%s chosen=%s',
            round3(ph),
            ph_ok,
            round3(hp),
            hp_ok,
            chosen,
        )
    return (ph_ok or hp_ok), chosen


def max_contra_sentence(nli, premise: str, hypothesis: str) -> float:
    best = 0.0
    for s in [s.strip() for s in re.split(SENT_SPLIT_RX, hypothesis) if s.strip()]:
        sc = nli.bidirectional_scores(premise, s)
        best = max(best, float(agg_max(sc).get('contradiction', 0.0)))
    return best
