# tests/test_integration_debate.py
import os
import re
import unicodedata

import pytest
from dotenv import load_dotenv

from app.infra.service import get_service  # used by _get_service_instance()

load_dotenv()

# If your server still returns "The debate has already ended.",
# change this constant accordingly.
END_MARKER = 'The debate has already ended.'

# ----------------------------
# Helpers
# ----------------------------


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
