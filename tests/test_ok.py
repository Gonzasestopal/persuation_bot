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
    want1 = _norm(f'Mantengámonos en el tema "{topic}" y en este idioma.')
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


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_never_conclude_after_five_turns_aligned_CON(client):
    """
    Topic: 'Dogs are humans' best friend' | Bot side: CON
    User sends 5 CON-aligned turns → debate should NOT conclude after 5 assistant replies.
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
        'Dogs are costly and time-consuming; many people cannot commit daily walks, training, and vet bills.',
        'Allergies and shedding make dogs impractical for many households; other pets or no pets can be healthier.',
        'Public safety and noise concerns matter; barking and bites disrupt neighbors more than most alternatives.',
        'Cats and other companions can offer affection with fewer demands, suiting busy owners better than dogs.',
        "Culturally, 'best friend' varies; for countless people, dogs create stress rather than support or productivity.",
    ]

    for t in user_msgs:
        r = client.post('/messages', json={'conversation_id': cid, 'message': t})
        assert r.status_code == 200, r.text
        bot_msg = _last_bot_msg(r.json())
        assert bot_msg and bot_msg.strip()
        assert END_MARKER not in bot_msg, f'Unexpected early end: {bot_msg!r}'

    svc = _get_service_instance()
    state = svc.debate_store.get(conversation_id=cid)
    assert state is not None
    assert not getattr(state, 'match_concluded', False), (
        'Debate should not conclude when user aligns with CON.'
    )


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_never_conclude_after_five_turns_aligned_PRO(client):
    """
    Topic: 'Dogs are humans' best friend' | Bot side: PRO
    User sends 5 PRO-aligned turns → debate should NOT conclude after 5 assistant replies.
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
        'Dogs provide consistent companionship and reduce loneliness, improving mental health for many owners.',
        'Service and therapy dogs measurably assist people with disabilities and PTSD, showing unique, life-changing support.',
        'Daily walks and play increase owners’ physical activity, improving cardiovascular health and routine discipline.',
        'Dogs’ social facilitation helps people meet neighbors and build community, strengthening social ties and belonging.',
        'Their loyalty and attunement to human cues make them exceptionally responsive friends compared to most animals.',
    ]

    for t in user_msgs:
        r = client.post('/messages', json={'conversation_id': cid, 'message': t})
        assert r.status_code == 200, r.text
        bot_msg = _last_bot_msg(r.json())
        assert bot_msg and bot_msg.strip()
        assert END_MARKER not in bot_msg, f'Unexpected early end: {bot_msg!r}'

    svc = _get_service_instance()
    state = svc.debate_store.get(conversation_id=cid)
    assert state is not None
    assert not getattr(state, 'match_concluded', False), (
        'Debate should not conclude when user aligns with PRO.'
    )
