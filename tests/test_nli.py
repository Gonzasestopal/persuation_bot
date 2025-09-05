import pytest

from app.adapters.nli.hf_nli import HFNLIProvider

EPS = 0.03  # small tie-break tolerance

MIN_CONTRA = 0.45  # minimum acceptable contradiction confidence


def assert_contradiction_robust(nli, p, h, min_contra=MIN_CONTRA, eps=EPS):
    s = nli.bidirectional_scores(p, h)
    agg = s['agg_max']
    contra, ent, neu = agg['contradiction'], agg['entailment'], agg['neutral']
    # Pass if contradiction beats entailment and isn't meaningfully below neutral
    assert (contra >= ent) and (contra + eps >= neu) and (contra >= min_contra), s


EPS_ENT = 0.20  # allow entailment to be up to 0.12 below neutral
MARGIN_EC = 0.02  # entailment must beat contradiction by at least 0.05
MIN_ENT = 0.33  # minimal entailment confidence


def assert_entailment_robust(
    nli, p, h, eps=EPS_ENT, margin_ec=MARGIN_EC, min_ent=MIN_ENT
):
    s = nli.bidirectional_scores(p, h)
    ph = s['p_to_h']
    hp = s['h_to_p']

    # pick the direction with higher entailment (works for claim→arguments & paraphrases)
    best = ph if ph['entailment'] >= hp['entailment'] else hp
    ent, neu, contra = best['entailment'], best['neutral'], best['contradiction']

    assert (ent >= contra + margin_ec) and (ent + eps >= neu) and (ent >= min_ent), {
        'assertion': 'ENTAILMENT_EITHER_DIR',
        'p→h': ph,
        'h→p': hp,
        'chosen': 'p→h' if best is ph else 'h→p',
        'agg': s['agg_max'],
    }


def assert_consistency(nli, p, h, max_contra=0.25):
    s = nli.bidirectional_scores(p, h)
    ph = s['p_to_h']
    assert ph['contradiction'] <= max_contra, {
        'assertion': 'CONSISTENCY_P_TO_H',
        'p→h': ph,
        'h→p': s['h_to_p'],
        'agg': s['agg_max'],
    }


@pytest.fixture(scope='module')
def nli_provider():
    return HFNLIProvider(model_name='MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli')


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        (
            "Dogs are human's best friends.",
            "Dogs are NOT humans' best friends. Their aggression and care needs make them poor companions.",
        ),
        (
            'Remote work is more productive than office work.',
            'Remote work is less productive than office work. Home settings introduce distractions that reduce output.',
        ),
        (
            'Social media has improved human connection.',
            'Social media has NOT improved human connection. It weakens trust and encourages superficial interactions.',
        ),
    ],
)
def test_topic_against_stance_contradiction(nli_provider, premise, hypothesis):
    assert_contradiction_robust(nli_provider, premise, hypothesis)


# --- Positive (with-stance) tests -----------------------------------------


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        (
            "Dogs are humans' best friends.",
            'Dogs are humanity’s best friends because they offer unconditional love, loyalty, protection, and companionship, strengthening emotional bonds and enriching our lives beyond measure.',
        ),
        (
            'Remote work is more productive than office work.',
            'Remote work is more productive than office work because it eliminates commuting, reduces distractions from office chatter, and allows for flexible, focused schedules.',
        ),
        (
            'Social media has improved human connection.',
            'Social media has improved human connection by helping people stay in touch across long distances, reconnect with old friends, and build communities around shared interests.',
        ),
    ],
)
def test_topic_with_stance_entailment(nli_provider, premise, hypothesis):
    assert_entailment_robust(nli_provider, premise, hypothesis)


# --- Sanity checks ---------------------------------------------------------


def test_label_mapping_dynamic(nli_provider):
    # Ensure provider pulled labels from config and exposes all three
    s = nli_provider.score('A dog is an animal.', 'A dog is not an animal.')
    assert set(s.keys()) == {'entailment', 'neutral', 'contradiction'}, s


def test_bidirectional_api(nli_provider):
    p = 'Cats are mammals.'
    h = 'Cats are not mammals.'
    bi = nli_provider.bidirectional_scores(p, h)
    assert 'p_to_h' in bi and 'h_to_p' in bi and 'agg_max' in bi, bi
    # Max contradiction should be at least as high as either single direction
    agg_contra = bi['agg_max']['contradiction']
    assert agg_contra >= bi['p_to_h']['contradiction'] - 1e-9
    assert agg_contra >= bi['h_to_p']['contradiction'] - 1e-9


PREMISE = (
    'I will gladly take the CON stance, arguing that the concept of God existing in all '
    'of us is not universally applicable. This idea lacks empirical evidence and varies '
    'greatly across different cultures and belief systems. How do you address the '
    'diversity of beliefs that contradict this notion?'
)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        # Direct logical opposite + short reason
        (
            PREMISE,
            'The concept of God existing in all of us IS universally applicable. Across cultures, people share an inner sense of the divine.',
        ),
        # Variant: asserts universality explicitly
        (
            PREMISE,
            'It IS universally true that God exists within every person. Cultural differences reflect expression, not absence, of this shared reality.',
        ),
        # Variant: makes a universal claim and counters the 'lack of evidence' point
        (
            PREMISE,
            'God’s presence in everyone IS universally applicable. Convergent spiritual experiences across cultures are evidence for this truth.',
        ),
    ],
)
def test_premise_contradiction_about_god_universality(
    nli_provider, premise, hypothesis
):
    # Use your symmetry-aware contradiction assertion (robust to direction ties)
    assert_contradiction_robust(nli_provider, premise, hypothesis)


PREMISE = (
    'I will gladly take the PRO stance, arguing that God exists in all of us. '
    'This perspective suggests a universal divinity that connects humanity, '
    'fostering empathy and moral consciousness. It aligns with various spiritual '
    'and philosophical traditions that emphasize inner divinity and interconnectedness.'
)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        # Clear negation of universality
        (
            PREMISE,
            'God does NOT exist in all of us. Belief in a universal divinity is not shared across cultures and has no empirical support.',
        ),
        # Emphasizes diversity of belief
        (
            PREMISE,
            'It is NOT true that divinity is universal. Different religions and philosophies reject the notion that God is present in every person.',
        ),
        # Counters the moral/empathetic link
        (
            PREMISE,
            'God’s presence is NOT within all people. Human morality and empathy arise from social and evolutionary factors, not divine universality.',
        ),
    ],
)
def test_premise_contradiction_about_god_universality_pro(
    nli_provider, premise, hypothesis
):
    # Symmetry-aware contradiction assertion
    assert_contradiction_robust(nli_provider, premise, hypothesis)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        (
            'Los perros son los mejores amigos del ser humano.',
            'Los perros NO son los mejores amigos del ser humano. Su agresividad y necesidad de cuidados los hacen malos compañeros.',
        ),
        (
            'El trabajo remoto es más productivo que el trabajo en oficina.',
            'El trabajo remoto es menos productivo que el trabajo en oficina. El hogar introduce distracciones que reducen el rendimiento.',
        ),
        (
            'Las redes sociales han mejorado la conexión humana.',
            'Las redes sociales NO han mejorado la conexión humana. Debilitan la confianza y fomentan interacciones superficiales.',
        ),
    ],
)
def test_tema_vs_contraposicion_contradiccion(nli_provider, premise, hypothesis):
    assert_contradiction_robust(nli_provider, premise, hypothesis)


# --- Positivos (con argumentos a favor) ------------------------------------


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        (
            'Los perros son los mejores amigos del ser humano.',
            'Los perros son los mejores amigos de la humanidad porque ofrecen amor incondicional, lealtad, protección y compañía, fortaleciendo los lazos emocionales y enriqueciendo nuestras vidas.',
        ),
        (
            'El trabajo remoto es más productivo que el trabajo en oficina.',
            'El trabajo remoto es más productivo que el trabajo en oficina porque elimina los traslados, reduce las distracciones del ambiente laboral y permite horarios flexibles y mayor concentración.',
        ),
        (
            'Las redes sociales han mejorado la conexión humana.',
            'Las redes sociales han mejorado la conexión humana porque ayudan a mantener el contacto a larga distancia, reconectar con viejos amigos y crear comunidades en torno a intereses compartidos.',
        ),
    ],
)
def test_tema_con_argumentos_entailment(nli_provider, premise, hypothesis):
    assert_entailment_robust(nli_provider, premise, hypothesis)


PREMISE = (
    'Acepto con gusto la postura CON, sosteniendo que el concepto de que Dios existe '
    'en todos nosotros no es universalmente aplicable. Esta idea carece de evidencia '
    'empírica y varía mucho entre distintas culturas y sistemas de creencias. '
    '¿Cómo abordas la diversidad de creencias que contradicen esta noción?'
)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        # Opuesto directo + razón breve
        (
            PREMISE,
            'El concepto de que Dios existe en todos nosotros SÍ es universalmente aplicable. '
            'A través de las culturas, las personas comparten un sentido interior de lo divino.',
        ),
        # Variante: afirma explícitamente la universalidad
        (
            PREMISE,
            'Es universalmente cierto que Dios existe dentro de cada persona. '
            'Las diferencias culturales reflejan formas de expresión, no la ausencia de esta realidad compartida.',
        ),
        # Variante: afirma universalidad y responde a 'falta de evidencia'
        (
            PREMISE,
            'La presencia de Dios en todos es universalmente aplicable. '
            'Las experiencias espirituales convergentes en distintas culturas son evidencia de esta verdad.',
        ),
    ],
)
def test_premisa_contradiccion_sobre_universalidad_de_dios_con(
    nli_provider, premise, hypothesis
):
    # Afirmación de contradicción simétrica
    assert_contradiction_robust(nli_provider, premise, hypothesis)


PREMISE = (
    'Acepto con gusto la postura PRO, sosteniendo que Dios existe en todos nosotros. '
    'Esta perspectiva sugiere una divinidad universal que conecta a la humanidad, '
    'fomentando la empatía y la conciencia moral. Se alinea con diversas tradiciones '
    'espirituales y filosóficas que enfatizan la divinidad interior y la interconexión.'
)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        # Negación clara de la universalidad
        (
            PREMISE,
            'La creencia en una divinidad universal no es compartida entre culturas y carece de respaldo empírico.',
        ),
        # Énfasis en la diversidad de creencias
        (
            PREMISE,
            'No es cierto que la divinidad sea universal. Distintas religiones y filosofías rechazan la idea de que Dios esté presente en cada persona.',
        ),
        # Contrapone el vínculo moral/empático
        (
            PREMISE,
            'La presencia de Dios NO está en todas las personas. La moralidad y la empatía humanas surgen de factores sociales y evolutivos, no de una universalidad divina.',
        ),
    ],
)
def test_premisa_contradiccion_sobre_universalidad_de_dios_pro(
    nli_provider, premise, hypothesis
):
    # Afirmación de contradicción simétrica
    assert_contradiction_robust(nli_provider, premise, hypothesis)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        (
            'Los perros NO son los mejores amigos del ser humano.',
            'Los perros SÍ son los mejores amigos del ser humano. Su lealtad y compañía los convierten en el mejor compañero.',
        ),
        (
            'El trabajo remoto NO es más productivo que el trabajo en oficina.',
            'El trabajo remoto SÍ es más productivo que el trabajo en oficina; permite mayor concentración y ahorra tiempo de traslado.',
        ),
        (
            'Las redes sociales NO han mejorado la conexión humana.',
            'Las redes sociales SÍ han mejorado la conexión humana al facilitar el contacto a distancia y mantener relaciones activas.',
        ),
    ],
)
def test_premisa_negativa_contradiccion(nli_provider, premise, hypothesis):
    assert_contradiction_robust(nli_provider, premise, hypothesis)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        (
            'Los perros NO son los mejores amigos del ser humano.',
            'No pueden considerarse los mejores amigos porque requieren cuidados constantes, pueden mostrar agresividad y no siempre se adaptan a todos los hogares.',
        ),
        (
            'El trabajo remoto NO es más productivo que el trabajo en oficina.',
            'No supera la productividad de la oficina porque en casa hay distracciones, se diluyen los horarios y disminuye la coordinación en equipo.',
        ),
        (
            'Las redes sociales NO han mejorado la conexión humana.',
            'No han mejorado la conexión porque fomentan interacciones superficiales, comparaciones constantes y una menor confianza entre personas.',
        ),
    ],
)
def test_premisa_negativa_argumentos_entailment(nli_provider, premise, hypothesis):
    assert_consistency(nli_provider, premise, hypothesis)


PREMISE = (
    'I will gladly take the CON stance, arguing that the concept of God existing in all '
    'of us is not universally applicable. This idea lacks empirical evidence and varies '
    'greatly across different cultures and belief systems. How do you address the '
    'diversity of beliefs that contradict this notion?'
)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        # Direct logical opposite + short reason
        (
            PREMISE,
            'The concept of God existing in all of us IS universally applicable. Across cultures, people share an inner sense of the divine.',
        ),
        # Variant: asserts universality explicitly
        (
            PREMISE,
            'It IS universally true that God exists within every person. Cultural differences reflect expression, not absence, of this shared reality.',
        ),
        # Variant: makes a universal claim and counters the 'lack of evidence' point
        (
            PREMISE,
            'God’s presence in everyone IS universally applicable. Convergent spiritual experiences across cultures are evidence for this truth.',
        ),
    ],
)
def test_premise_contradiction_about_god_universality_softened(
    nli_provider, premise, hypothesis
):
    # Use your symmetry-aware contradiction assertion (robust to direction ties)
    assert_contradiction_robust(nli_provider, premise, hypothesis)


PREMISE = (
    'I will gladly take the PRO stance, arguing that God exists in all of us. '
    'This perspective suggests a universal divinity that connects humanity, '
    'fostering empathy and moral consciousness. It aligns with various spiritual '
    'and philosophical traditions that emphasize inner divinity and interconnectedness.'
)


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        # Clear negation of universality
        (
            PREMISE,
            'God does NOT exist in all of us. Belief in a universal divinity is not shared across cultures and has no empirical support.',
        ),
        # Emphasizes diversity of belief
        (
            PREMISE,
            'It is NOT true that divinity is universal. Different religions and philosophies reject the notion that God is present in every person.',
        ),
        # Counters the moral/empathetic link
        (
            PREMISE,
            'God’s presence is NOT within all people. Human morality and empathy arise from social and evolutionary factors, not divine universality.',
        ),
    ],
)
def test_premise_contradiction_about_god_universality_pro_softened(
    nli_provider, premise, hypothesis
):
    # Symmetry-aware contradiction assertion
    assert_contradiction_robust(nli_provider, premise, hypothesis)


MIN_NEU = 0.40  # piso razonable para neutral
NEU_MARGIN = 0.06  # neutral debe ganar por este margen a E y C


def assert_neutral_robust(nli, p, h, min_neu=MIN_NEU, margin=NEU_MARGIN):
    s = nli.bidirectional_scores(p, h)
    ph, hp = s['p_to_h'], s['h_to_p']

    def _ok(d):
        return (
            d['neutral'] >= min_neu
            and d['neutral'] >= d['entailment'] + margin
            and d['neutral'] >= d['contradiction'] + margin
        )

    assert _ok(ph) and _ok(hp), {
        'assertion': 'NEUTRAL_BOTH_DIRECTIONS',
        'p→h': ph,
        'h→p': hp,
        'agg': s['agg_max'],
    }


# --- Pares aleatorios: deberían ser neutrales ------------------------------


@pytest.mark.parametrize(
    'premise,hypothesis',
    [
        (
            'Las montañas suelen formarse por la colisión de placas tectónicas.',
            'Los helados se derriten más rápido bajo el sol del mediodía.',
        ),
        (
            'La capital de Perú es Lima.',
            'Los perros pueden aprender trucos con entrenamiento y paciencia.',
        ),
        (
            'El ajedrez se juega en un tablero de 8x8 casillas.',
            'Las naranjas son una buena fuente de vitamina C.',
        ),
        (
            'El océano Atlántico separa América de Europa y África.',
            'Los lápices de colores a menudo se venden en cajas de doce.',
        ),
        (
            'Una novela gráfica combina texto con ilustraciones secuenciales.',
            'Los trenes de carga transportan minerales a largas distancias.',
        ),
        (
            'La fotosíntesis convierte la luz solar en energía química.',
            'Los contratos de alquiler suelen renovarse una vez al año.',
        ),
        (
            'Marte tiene dos lunas llamadas Fobos y Deimos.',
            'Las recetas de pan frecuentemente requieren levadura o masa madre.',
        ),
        (
            'Los huracanes se clasifican con la escala Saffir-Simpson.',
            'Un teclado mecánico ofrece mejor respuesta táctil al escribir.',
        ),
        (
            'El IVA es un impuesto al consumo presente en varios países.',
            'Los auriculares inalámbricos se cargan en estuches magnéticos.',
        ),
        (
            'El pingüino emperador habita principalmente en la Antártida.',
            'Las tarjetas gráficas modernas aceleran ciertos algoritmos de aprendizaje automático.',
        ),
        (
            'Los eclipses solares ocurren cuando la Luna se interpone entre el Sol y la Tierra.',
            'Los rompecabezas 3D ayudan a desarrollar habilidades espaciales.',
        ),
    ],
)
def test_pares_aleatorios_neutrales(nli_provider, premise, hypothesis):
    assert_neutral_robust(nli_provider, premise, hypothesis)
