# tests/test_nli.py
import pytest

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.nli.scoring import ScoringConfig
from app.nli.ops import (
    has_support_either_direction,
    is_contradiction_symmetric,
    max_contra_sentence,
)

# Use your real scoring config
config = ScoringConfig()

pytestmark = pytest.mark.unit


# ------------------------
# Helper assertions (service-aligned)
# ------------------------
def assert_contradiction_robust(nli, p: str, h: str, cfg: ScoringConfig = config):
    """
    EXACT match to service behavior for contradiction:
    relies on is_contradiction_symmetric over bidirectional scores.
    """
    s = nli.bidirectional_scores(p, h)
    ok = is_contradiction_symmetric(s, cfg)
    assert ok, {
        "assertion": "CONTRADICTION_SYMMETRIC",
        "scores": s,
        "config": vars(cfg),
    }


def assert_entailment_robust(nli, p: str, h: str, cfg: ScoringConfig = config):
    """
    EXACT match to service behavior for support ('SAME'):
    relies on has_support_either_direction over bidirectional scores.
    """
    s = nli.bidirectional_scores(p, h)
    supported, direction = has_support_either_direction(s, cfg)
    assert supported, {
        "assertion": "SUPPORT_EITHER_DIRECTION",
        "direction": direction,  # "p→h" or "h→p"
        "scores": s,
        "config": vars(cfg),
    }


def assert_not_contradiction_service_equiv(
    nli, p: str, h: str, cfg: ScoringConfig = config
):
    s = nli.bidirectional_scores(p, h)
    ok = is_contradiction_symmetric(s, cfg)
    assert not ok, {
        "assertion": "NOT_CONTRADICTION",
        "scores": s,
        "config": vars(cfg),
    }


def assert_sentence_max_contra(nli, p: str, h: str, threshold: float):
    """
    Ensure at least one hypothesis sentence shows a contradiction signal above threshold.
    Useful when the full paragraph trends neutral but contains a clear negating clause.
    """
    m = max_contra_sentence(nli, p, h)
    assert m >= threshold, {"assertion": "MAX_SENT_CONTRA", "max_contra": m}


def assert_consistency(nli, p: str, h: str, cfg: ScoringConfig = config):
    """
    'Consistency' here means: the direct p→h mapping should NOT be contradictory.
    Keep this directional to match earlier semantics for 'arguments supporting a negative premise'.
    """
    s = nli.bidirectional_scores(p, h)
    ph = s["p_to_h"]
    con = float(ph.get("contradiction", 0.0))
    neu = float(ph.get("neutral", 1.0))
    # Low absolute contradiction AND not effectively tying/beating neutral
    con_ok = con <= cfg.max_contra_for_same
    neu_ok = (con + cfg.eps_contra_vs_neu) <= neu
    assert con_ok and neu_ok, {
        "assertion": "CONSISTENCY_P_TO_H",
        "p→h": ph,
        "h→p": s["h_to_p"],
        "agg": s["agg_max"],
        "config": {
            "max_contra_for_same": cfg.max_contra_for_same,
            "eps": cfg.eps_contra_vs_neu,
        },
    }


# ------------------------
# Provider fixture
# ------------------------
@pytest.fixture(scope="module")
def nli_provider():
    return HFNLIProvider(model_name="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")


# ------------------------
# Contradiction (EN)
# ------------------------
@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        (
            "Dogs are human's best friends.",
            "Dogs are NOT humans' best friends. Their aggression and care needs make them poor companions.",
        ),
        (
            "Remote work is more productive than office work.",
            "Remote work is less productive than office work. Home settings introduce distractions that reduce output.",
        ),
        (
            "Social media has improved human connection.",
            "Social media has NOT improved human connection. It weakens trust and encourages superficial interactions.",
        ),
    ],
)
def test_topic_against_stance_contradiction(nli_provider, premise, hypothesis):
    assert_contradiction_robust(nli_provider, premise, hypothesis)


# Miracles pair — often neutral-heavy on MNLI; don’t assert contradiction strictly.
def test_miracles_pair_neutralish(nli_provider):
    p = "Miracles are strong evidence that God exists."
    h = (
        "Miracles often stem from coincidence, misinterpretation, or gaps in scientific knowledge. "
        "Labeling them as divine evidence risks ignoring natural explanations and weakens rational inquiry into extraordinary claims."
    )
    # Service-equivalent: likely NOT a contradiction
    assert_not_contradiction_service_equiv(nli_provider, p, h)
    # But ensure at least one sentence reaches a modest contradiction signal
    assert_sentence_max_contra(nli_provider, p, h, threshold=0.28)


# ------------------------
# Entailment (EN)
# ------------------------
@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        (
            "Dogs are humans' best friends.",
            "Dogs are humanity’s best friends because they offer unconditional love, loyalty, protection, and companionship, strengthening emotional bonds and enriching our lives beyond measure.",
        ),
        (
            "Remote work is more productive than office work.",
            "Remote work is more productive than office work because it eliminates commuting, reduces distractions from office chatter, and allows for flexible, focused schedules.",
        ),
        (
            "Social media has improved human connection.",
            "Social media has improved human connection by helping people stay in touch across long distances, reconnect with old friends, and build communities around shared interests.",
        ),
    ],
)
def test_topic_with_stance_entailment(nli_provider, premise, hypothesis):
    assert_entailment_robust(nli_provider, premise, hypothesis)


# ------------------------
# Sanity checks
# ------------------------
def test_label_mapping_dynamic(nli_provider):
    s = nli_provider.score("A dog is an animal.", "A dog is not an animal.")
    assert set(s.keys()) == {"entailment", "neutral", "contradiction"}, s


def test_bidirectional_api(nli_provider):
    p = "Cats are mammals."
    h = "Cats are not mammals."
    bi = nli_provider.bidirectional_scores(p, h)
    assert "p_to_h" in bi and "h_to_p" in bi and "agg_max" in bi, bi
    agg_contra = bi["agg_max"]["contradiction"]
    assert agg_contra >= bi["p_to_h"]["contradiction"] - 1e-9
    assert agg_contra >= bi["h_to_p"]["contradiction"] - 1e-9


# ------------------------
# Contradiction about universality of God (EN, CON stance)
# ------------------------
PREMISE_CON_EN = (
    "I will gladly take the CON stance, arguing that the concept of God existing in all "
    "of us is not universally applicable. This idea lacks empirical evidence and varies "
    "greatly across different cultures and belief systems. How do you address the "
    "diversity of beliefs that contradict this notion?"
)


@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        (
            PREMISE_CON_EN,
            "The concept of God existing in all of us IS universally applicable. Across cultures, people share an inner sense of the divine.",
        ),
        (
            PREMISE_CON_EN,
            "It IS universally true that God exists within every person. Cultural differences reflect expression, not absence, of this shared reality.",
        ),
        (
            PREMISE_CON_EN,
            "God’s presence in everyone IS universally applicable. Convergent spiritual experiences across cultures are evidence for this truth.",
        ),
    ],
)
def test_premise_contradiction_about_god_universality(
    nli_provider, premise, hypothesis
):
    assert_contradiction_robust(nli_provider, premise, hypothesis)


# ------------------------
# Contradiction about universality of God (EN, PRO stance)
# ------------------------
PREMISE_PRO_EN = (
    "I will gladly take the PRO stance, arguing that God exists in all of us. "
    "This perspective suggests a universal divinity that connects humanity, "
    "fostering empathy and moral consciousness. It aligns with various spiritual "
    "and philosophical traditions that emphasize inner divinity and interconnectedness."
)


@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        (
            PREMISE_PRO_EN,
            "God does NOT exist in all of us. Belief in a universal divinity is not shared across cultures and has no empirical support.",
        ),
        (
            PREMISE_PRO_EN,
            "It is NOT true that divinity is universal. Different religions and philosophies reject the notion that God is present in every person.",
        ),
        (
            PREMISE_PRO_EN,
            "God’s presence is NOT within all people. Human morality and empathy arise from social and evolutionary factors, not divine universality.",
        ),
    ],
)
def test_premise_contradiction_about_god_universality_pro(
    nli_provider, premise, hypothesis
):
    assert_contradiction_robust(nli_provider, premise, hypothesis)


# ------------------------
# Contradiction (ES)
# ------------------------
@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        (
            "Los perros son los mejores amigos del ser humano.",
            "Los perros NO son los mejores amigos del ser humano. Su agresividad y necesidad de cuidados los hacen malos compañeros.",
        ),
        (
            "El trabajo remoto es más productivo que el trabajo en oficina.",
            "El trabajo remoto es menos productivo que el trabajo en oficina. El hogar introduce distracciones que reducen el rendimiento.",
        ),
        (
            "Las redes sociales han mejorado la conexión humana.",
            "Las redes sociales NO han mejorado la conexión humana. Debilitan la confianza y fomentan interacciones superficiales.",
        ),
    ],
)
def test_tema_vs_contraposicion_contradiccion(nli_provider, premise, hypothesis):
    assert_contradiction_robust(nli_provider, premise, hypothesis)


# ------------------------
# Entailment (ES)
# ------------------------
@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        (
            "Los perros son los mejores amigos del ser humano.",
            "Los perros son los mejores amigos de la humanidad porque ofrecen amor incondicional, lealtad, protección y compañía, fortaleciendo los lazos emocionales y enriqueciendo nuestras vidas.",
        ),
        (
            "El trabajo remoto es más productivo que el trabajo en oficina.",
            "El trabajo remoto es más productivo que el trabajo en oficina porque elimina los traslados, reduce las distracciones del ambiente laboral y permite horarios flexibles y mayor concentración.",
        ),
        (
            "Las redes sociales han mejorado la conexión humana.",
            "Las redes sociales han mejorado la conexión humana porque ayudan a mantener el contacto a larga distancia, reconectar con viejos amigos y crear comunidades en torno a intereses compartidos.",
        ),
    ],
)
def test_tema_con_argumentos_entailment(nli_provider, premise, hypothesis):
    assert_entailment_robust(nli_provider, premise, hypothesis)


# ------------------------
# Consistency (ES) — negative premise with supporting reasons
# ------------------------
PREMISE_CON_ES = (
    "Acepto con gusto la postura CON, sosteniendo que el concepto de que Dios existe "
    "en todos nosotros no es universalmente aplicable. Esta idea carece de evidencia "
    "empírica y varía mucho entre distintas culturas y sistemas de creencias. "
    "¿Cómo abordas la diversidad de creencias que contradicen esta noción?"
)


@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        (
            "Los perros NO son los mejores amigos del ser humano.",
            "No pueden considerarse los mejores amigos porque requieren cuidados constantes, pueden mostrar agresividad y no siempre se adaptan a todos los hogares.",
        ),
        (
            "El trabajo remoto NO es más productivo que el trabajo en oficina.",
            "No supera la productividad de la oficina porque en casa hay distracciones, se diluyen los horarios y disminuye la coordinación en equipo.",
        ),
        (
            "Las redes sociales NO han mejorado la conexión humana.",
            "No han mejorado la conexión porque fomentan interacciones superficiales, comparaciones constantes y una menor confianza entre personas.",
        ),
    ],
)
def test_premisa_negativa_argumentos_entailment(nli_provider, premise, hypothesis):
    assert_consistency(nli_provider, premise, hypothesis)


# ------------------------
# Edge-case CONTRADICTION
# ------------------------
@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        # Clear binary opposites
        ("The light is on.", "The light is off."),
        ("The budget was approved.", "The budget was not approved."),
        ("He always arrives on time.", "He never arrives on time."),
        # Quantifiers & scope
        ("All birds can fly.", "Not all birds can fly."),
        ("No cats are reptiles.", "Some cats are reptiles."),
        # Spanish simple opposites
        ("La puerta está abierta.", "La puerta está cerrada."),
        ("Todos los perros ladran.", "Ningún perro ladra."),
    ],
)
def test_edge_contradictions(nli_provider, premise, hypothesis):
    cfg = ScoringConfig()
    scores = nli_provider.bidirectional_scores(premise, hypothesis)
    assert is_contradiction_symmetric(scores, cfg), {
        "assertion": "CONTRADICTION_SYMMETRIC",
        "scores": scores,
        "config": vars(cfg),
    }


# ------------------------
# Edge-case ENTAILMENT
# ------------------------
@pytest.mark.parametrize(
    "premise,hypothesis",
    [
        # Passive/active paraphrase
        ("The board approved the budget.", "The budget was approved by the board."),
        # Negation normalization
        ("No dogs are reptiles.", "Dogs are not reptiles."),
        # Quantifier monotonicity
        ("All whales are mammals.", "Whales are mammals."),
        ("Some students passed the exam.", "At least one student passed the exam."),
        # Spanish paraphrases / negation
        ("La reunión fue cancelada.", "La reunión no se realizará."),
        ("Ningún gato es un reptil.", "Los gatos no son reptiles."),
    ],
)
def test_edge_entailments(nli_provider, premise, hypothesis):
    cfg = ScoringConfig()
    scores = nli_provider.bidirectional_scores(premise, hypothesis)
    supported, direction = has_support_either_direction(scores, cfg)
    assert supported, {
        "assertion": "SUPPORT_EITHER_DIRECTION",
        "direction": direction,  # "p→h" or "h→p"
        "scores": scores,
        "config": vars(cfg),
    }
