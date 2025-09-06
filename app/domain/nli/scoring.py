from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    # Contradiction thresholds
    contradiction_threshold: float = 0.60
    strict_contra_threshold: float = 0.85
    contradiction_threshold_soft: float = 0.45
    eps_contra_vs_neu: float = 0.03
    min_delta_con_neu: float = 0.12

    # Entailment thresholds
    min_ent_for_same: float = 0.70
    margin_ent_vs_neu: float = 0.25
    max_contra_for_same: float = 0.40
    margin_ec: float = 0.02
    eps_ent: float = 0.20
    sentence_probe_min: float = 0.28  # ensure this exists

    # On-topic filters
    topic_signal_min: float = 0.30
    topic_neu_max: float = 0.72

    # Misc
    min_user_words: int = 8
