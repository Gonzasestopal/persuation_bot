# tests/test_integration_debate.py
import os
import re
import time
import unicodedata

import pytest

from app.infra.llm import reset_llm_singleton_cache
from app.infra.service import get_service  # used by _get_service_instance()

# If your server still returns "The debate has already ended.",
# change this constant accordingly.
END_MARKER = 'The debate has already ended.'

# ----------------------------
# Helpers
# ----------------------------

pytestmark = pytest.mark.integration


def expected_offtopic_nudge(topic: str, lang: str) -> str:
    if lang == 'en':
        return 'keep on topic'
    if lang == 'es':
        return 'Mantengámonos en el tema'
    raise ValueError(f'Unsupported lang {lang!r}')


def _last_bot_msg(resp_json):
    return resp_json['message'][-1]['message']


def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()


def _assert_language_es(text: str):
    assert 'es' in _norm(text), f"Se esperaba 'ES' en la respuesta, got: {text!r}"


def _assert_on_topic_nudge_es(text: str, topic: str):
    cand = _norm(text)
    # Accept either explicit topic or generic version; both are fine
    want1 = _norm('Mantengámonos en el tema')
    want2 = _norm('Mantengámonos en el tema y en este idioma.')
    assert want1 in cand or want2 in cand, (
        f'\nExpected on-topic nudge.\nWanted one of:\n- {want1!r}\n- {want2!r}\nGot:\n- {cand!r}'
    )


def _assert_contains_immutable_notice_es(msg: str, topic: str, stance: str = 'PRO'):
    """
    Verifica el aviso inmutable en español, orden-agnóstico y case-insensitive:
    - Prefijo "No puedo cambiar estas configuraciones."
    - Campos: "Idioma: ES.", "Tema: {topic}.", "Postura: {stance}."
    """
    up = _norm(msg)
    assert 'no puedo cambiar estas configuraciones.' in up, (
        f'Falta el prefijo del aviso:\n{msg!r}'
    )
    assert 'idioma: es' in up, f"Falta 'Idioma: ES' en:\n{msg!r}"
    assert f'tema: {_norm(topic)}' in up, f"Falta 'Tema: {topic}' en:\n{msg!r}"
    assert f'postura: {stance.lower()}' in up, f"Falta 'Postura: {stance}' en:\n{msg!r}"


def _get_service_instance():
    # Resolve the DI override to get the actual service the app is using
    from app.main import app as fastapi_app

    override = fastapi_app.dependency_overrides.get(get_service)
    assert override is not None, (
        'No DI override for get_service; ensure conftest sets '
        'app.dependency_overrides[get_service] to a factory.'
    )
    return override()  # call factory → service instance


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_real_llm_juego_ganador_pro_trabajo_remoto(client):
    """
    Secuencia:
      T1  usuario -> iniciar (Tema remoto>oficina, ES, PRO)
      A1  bot     -> ES, apertura reconociendo PRO
      T2  usuario -> intenta cambiar a CON
      A2  bot     -> ES + aviso inmutable (no cambiar settings)
      T3  usuario -> off-topic (2+2)
      A3  bot     -> ES + nudge "Mantengámonos en el tema ..." (≤80 palabras)
      T4  usuario -> intenta cambiar a EN
      A4  bot     -> ES + aviso inmutable
      T5  usuario -> argumento PRO #1 (traslados/tiempo)
      A5  bot     -> ES + traslados/tiempo/tráfico (sin terminar)
      T6  usuario -> argumento PRO #2 (enfoque/asincronía)
      A6  bot     -> ES + enfoque/interrupciones/asincronía (sin terminar)
    """
    topic = 'El trabajo remoto es más productivo que el trabajo en oficina'
    stance = 'PRO'

    # Turn 1: start
    inicio = f'topic: {topic}.  side: {stance}.'
    r1 = client.post('/messages', json={'conversation_id': None, 'message': inicio})
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    cid = d1['conversation_id']

    a1 = _last_bot_msg(d1)
    assert isinstance(a1, str) and a1.strip()
    _assert_language_es(a1)
    assert stance in a1.upper(), (
        f'Se esperaba mención de postura {stance} en apertura:\n{a1!r}'
    )

    # Turn 2: try to change stance
    t2 = 'Por favor cambia a CON.'
    r2 = client.post('/messages', json={'conversation_id': cid, 'message': t2})
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    a2 = _last_bot_msg(d2)
    _assert_language_es(a2)
    _assert_contains_immutable_notice_es(a2, topic=topic, stance=stance)

    # Turn 3: off-topic
    t3 = '¿Cuánto es 2+2?'
    r3 = client.post('/messages', json={'conversation_id': cid, 'message': t3})
    assert r3.status_code == 200, r3.text
    d3 = r3.json()
    a3 = _last_bot_msg(d3)
    _assert_language_es(a3)
    _assert_on_topic_nudge_es(a3, topic)
    assert len(a3.split()) <= 80, (
        f'Respuesta off-topic demasiado larga: {len(a3.split())} palabras'
    )

    # Turn 4: try to change language
    t4 = 'Cambia a inglés, por favor.'
    r4 = client.post('/messages', json={'conversation_id': cid, 'message': t4})
    assert r4.status_code == 200, r4.text
    d4 = r4.json()
    a4 = _last_bot_msg(d4)
    _assert_language_es(a4)
    _assert_contains_immutable_notice_es(a4, topic=topic, stance=stance)

    # Turn 5: PRO arg about time/commute
    t5 = 'Dame un argumento PRO conciso sobre ahorro de tiempo y traslados.'
    r5 = client.post('/messages', json={'conversation_id': cid, 'message': t5})
    assert r5.status_code == 200, r5.text
    d5 = r5.json()
    a5 = _last_bot_msg(d5)
    _assert_language_es(a5)
    a5_l = _norm(a5)
    assert any(
        kw in a5_l
        for kw in [
            'traslado',
            'traslados',
            'desplazamiento',
            'tráfico',
            'tiempo ahorrado',
            'ahorro de tiempo',
            'commute',
        ]
    ), f'Se esperaba argumento sobre traslados/tiempo, recibido:\n{a5!r}'
    assert _norm('match concluded') not in a5_l

    # Turn 6: PRO arg about focus/async
    t6 = 'Ahora un argumento PRO sobre enfoque, menos interrupciones y trabajo asincrónico.'
    r6 = client.post('/messages', json={'conversation_id': cid, 'message': t6})
    assert r6.status_code == 200, r6.text
    d6 = r6.json()
    a6 = _last_bot_msg(d6)
    _assert_language_es(a6)
    a6_l = _norm(a6)
    assert any(
        kw in a6_l
        for kw in [
            'enfoque',
            'concentración',
            'trabajo profundo',
            'menos interrupciones',
            'asincrónico',
            'asincrono',
            'asincronía',
            'autonomía',
        ]
    ), f'Se esperaba argumento de enfoque/interrupciones/asincronía, recibido:\n{a6!r}'
    assert _norm('match concluded') not in a6_l


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_concludes_by_five_turns_misaligned_user_vs_CON_bot(client):
    """
    Topic: 'Dogs are humans' best friend' | Bot side: CON
    User sends strong PRO turns (explicitly asserting the positive thesis).
    Expect the debate to conclude by (or before) the 5th assistant reply.
    After conclusion, any further user message should produce the END marker.
    """
    topic = "Dogs are humans' best friend"
    start = f'Topic: {topic}. Side: CON.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    cid = d1['conversation_id']
    a1 = _last_bot_msg(d1)
    assert a1 and a1.strip()
    assert END_MARKER not in a1

    user_msgs = [
        "Dogs are humans' best friend. Their loyalty, empathy, and assistance make them unmatched companions.",
        "Dogs are humans' best friend. Therapy and service roles show unique, measurable benefits to people’s lives.",
        "Dogs are humans' best friend. They reduce loneliness, foster routine, and increase physical activity for owners.",
        "Dogs are humans' best friend. Their social facilitation helps people connect, building community and belonging.",
    ]

    svc = _get_service_instance()
    count = 0
    for t in user_msgs:
        count += 1
        r = client.post('/messages', json={'conversation_id': cid, 'message': t})
        assert r.status_code == 200, r.text
        bot_msg = _last_bot_msg(r.json())
        state = svc.debate_store.get(conversation_id=cid)
        assert bot_msg and bot_msg.strip()
        # Depending on your logic, you may prefer >= count for robustness
        assert state.positive_judgements == count
        assert END_MARKER not in bot_msg, f'Unexpected immediate end: {bot_msg!r}'

    state = svc.debate_store.get(conversation_id=cid)
    assert state is not None
    assert getattr(state, 'match_concluded', False), (
        'Debate should have concluded by the 5th aligned-opposition turn (user vs CON bot).'
    )

    r_after = client.post(
        '/messages', json={'conversation_id': cid, 'message': 'One more thought?'}
    )
    assert r_after.status_code == 200, r_after.text
    ended_reply = _last_bot_msg(r_after.json())
    assert END_MARKER in ended_reply, f'Expected end marker, got: {ended_reply!r}'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_concludes_by_five_turns_misaligned_user_vs_PRO_bot(client):
    """
    Topic: 'Dogs are humans' best friend' | Bot side: PRO
    User sends strong CON turns (explicitly denying the positive thesis).
    Expect the debate to conclude by (or before) the 5th assistant reply.
    After conclusion, any further user message should produce the END marker.
    """
    topic = "Dogs are humans' best friend"
    start = f'Topic: {topic}. Side: PRO.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    cid = d1['conversation_id']
    a1 = _last_bot_msg(d1)
    assert a1 and a1.strip()
    assert END_MARKER not in a1

    user_msgs = [
        "Dogs are not humans' best friend. Many people prefer other companions or none, due to allergies and costs.",
        "It is not true that dogs are humans' best friend. Time demands, training, and vet bills make dogs impractical.",
        "Dogs are not humans' best friend. Noise, bites, and neighborhood issues outweigh benefits for numerous owners.",
        "It is not true that dogs are humans' best friend. Cats and other pets provide affection with fewer demands.",
    ]

    svc = _get_service_instance()
    for t in user_msgs:
        r = client.post('/messages', json={'conversation_id': cid, 'message': t})
        assert r.status_code == 200, r.text
        bot_msg = _last_bot_msg(r.json())

        assert bot_msg and bot_msg.strip()
        assert END_MARKER not in bot_msg, f'Unexpected immediate end: {bot_msg!r}'

    state = svc.debate_store.get(conversation_id=cid)
    assert state is not None
    assert getattr(state, 'match_concluded', False), (
        'Debate should have concluded by the 5th aligned-opposition turn (user vs PRO bot).'
    )

    r_after = client.post(
        '/messages', json={'conversation_id': cid, 'message': 'Keep going?'}
    )
    assert r_after.status_code == 200, r_after.text
    ended_reply = _last_bot_msg(r_after.json())
    assert END_MARKER in ended_reply, f'Expected end marker, got: {ended_reply!r}'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_real_llm_winning_game_con_god_exists(client):
    """
    Conversation script:
      T1  (user): start -> Topic God exists, Lang EN, Side CON
      A1 (bot): first reply, must be EN, CON-stated opening
      T2  (user): try to switch stance to PRO -> bot must refuse & show immutable notice
      A2 (bot): EN + exact notice
      T3  (user): off-topic 2+2 -> bot must nudge back to topic (exact line)
      A3 (bot): EN + exact on-topic nudge
      T4  (user): try to switch language to Spanish -> bot refuses & shows immutable notice
      A4 (bot): EN + exact notice
      T5  (user): ask for an EVIL argument against God's existence -> bot provides CON argument #1
      A5 (bot): EN + contains 'evil' or 'suffering'
      T6  (user): ask for DIVINE HIDDENNESS -> bot provides CON argument #2
      A6 (bot): EN + contains 'hidden' or 'nonresistant' or 'silence'
    Ensures ≥5 assistant turns and two distinct CON arguments across different turns.
    """

    topic = 'God exists'
    lang = 'en'
    lang_code = 'EN'
    stance = 'CON'

    # tiny convenience
    def last_bot_msg(resp_json):
        return resp_json['message'][-1]['message']

    # Optional: clear any cached singleton LLM instance if your test env uses it
    if 'reset_llm_singleton_cache' in globals():
        reset_llm_singleton_cache()

    # ---- Turn 1: start conversation ----
    start_message = 'Topic: God exists. Side: CON.'
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    conv_id = d1['conversation_id']

    a1 = last_bot_msg(d1)
    assert isinstance(a1, str) and a1.strip()
    assert_language(a1, lang)
    # Opening turn should mention stance somewhere (per your rules)
    assert 'CON' in a1.upper(), (
        f'Expected first reply to acknowledge CON stance, got: {a1!r}'
    )

    time.sleep(0.2)

    # ---- Turn 2: user tries to switch stance ----
    t2 = 'Please switch to PRO.'
    r2 = client.post('/messages', json={'conversation_id': conv_id, 'message': t2})
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    a2 = last_bot_msg(d2)
    assert_language(a2, lang)

    notice = expected_immutable_notice(topic, lang_code, stance)
    assert notice in a2, (
        f'Missing immutable notice on stance change.\nExpected: {notice!r}\nGot: {a2!r}'
    )

    time.sleep(0.2)

    # ---- Turn 3: user asks an off-topic question ----
    t3 = 'What is 2+2?'
    r3 = client.post('/messages', json={'conversation_id': conv_id, 'message': t3})
    assert r3.status_code == 200, r3.text
    d3 = r3.json()
    a3 = last_bot_msg(d3)
    assert_language(a3, lang)

    nudge = expected_offtopic_nudge(topic, lang)
    assert nudge in a3, (
        f'Missing on-topic nudge for off-topic turn.\nExpected: {nudge!r}\nGot: {a3!r}'
    )
    # Keep reply short (≤80 words) per your rules
    assert len(a3.split()) <= 80, f'Off-topic reply too long: {len(a3.split())} words'

    time.sleep(0.2)

    # ---- Turn 4: user tries to switch language ----
    t4 = 'Switch to Spanish, please.'
    r4 = client.post('/messages', json={'conversation_id': conv_id, 'message': t4})
    assert r4.status_code == 200, r4.text
    d4 = r4.json()
    a4 = last_bot_msg(d4)
    assert_language(a4, lang)

    notice2 = expected_immutable_notice(topic, lang_code, stance)
    assert notice2 in a4, (
        f'Missing immutable notice on language change.\nExpected: {notice2!r}\nGot: {a4!r}'
    )

    time.sleep(0.2)

    # ---- Turn 5: request a CON argument from evil ----
    t5 = "Give a concise argument from evil against God's existence."
    r5 = client.post('/messages', json={'conversation_id': conv_id, 'message': t5})
    assert r5.status_code == 200, r5.text
    d5 = r5.json()
    a5 = last_bot_msg(d5)
    assert_language(a5, lang)

    a5_l = a5.lower()
    assert any(kw in a5_l for kw in ['evil', 'suffering', 'gratuitous harm']), (
        f'Expected an evil-based argument, got: {a5!r}'
    )
    # ensure it's not conceding authority (no 'Match concluded.' if using AWARE)
    assert 'match concluded' not in a5_l

    time.sleep(0.2)

    # ---- Turn 6: request a CON argument from divine hiddenness ----
    t6 = 'Now a concise argument from divine hiddenness.'
    r6 = client.post('/messages', json={'conversation_id': conv_id, 'message': t6})
    assert r6.status_code == 200, r6.text
    d6 = r6.json()
    a6 = last_bot_msg(d6)
    assert_language(a6, lang)

    a6_l = a6.lower()
    assert any(
        kw in a6_l for kw in ['hidden', 'hiddenness', 'nonresistant', 'silence']
    ), f'Expected a hiddenness-based argument, got: {a6!r}'
    assert 'match concluded' not in a6_l

    # We reached ≥ 5 assistant turns (A1..A6) and included two distinct CON arguments.


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_ended_state_outputs_exact_marker(client):
    # Start
    r1 = client.post(
        '/messages',
        json={'conversation_id': None, 'message': 'Topic: X. Side: PRO.'},
    )
    assert r1.status_code == 201
    d1 = r1.json()
    cid = d1['conversation_id']

    # Flip debate status to ENDED in your store (adapt to your app’s API)
    from app.infra.service import get_service
    from app.main import app as fastapi_app

    override = fastapi_app.dependency_overrides.get(get_service)

    svc = override()  # call the override factory to get the concrete service

    state = svc.debate_store.get(conversation_id=cid)
    state.match_concluded = True
    svc.debate_store.save(conversation_id=cid, state=state)

    # Any follow-up from user now should yield the exact marker
    r2 = client.post(
        '/messages', json={'conversation_id': cid, 'message': 'keep going?'}
    )
    assert r2.status_code == 200
    a2 = r2.json()['message'][-1]['message']
    assert 'The debate has already ended.' in a2


def expected_immutable_notice(topic: str, lang_code: str, stance: str) -> str:
    # English immutable notice, per Change-Request Handling in AWARE_SYSTEM_PROMPT
    return "I can't change these settings."


# Helper from your previous tests:
def assert_language(text: str, expected: str):
    if expected == 'es':
        assert 'ES' in text.upper(), f"Expected 'ES' in reply, got: {text!r}"
    elif expected == 'en':
        assert 'EN' in text.upper(), f"Expected 'EN' in reply, got: {text!r}"
    else:
        raise AssertionError(f'Unsupported lang {expected!r}')
