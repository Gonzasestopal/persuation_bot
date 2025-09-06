import os
import re
import time

import pytest

from app.adapters.llm.constants import Provider
from app.infra.llm import reset_llm_singleton_cache
from app.infra.service import get_service
from app.main import app
from app.settings import settings


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
@pytest.mark.parametrize(
    'start_message, expected_stance, second_msg, second_expected_stance',
    [
        (
            'Topic: Sports build character. Side: PRO.',
            'PRO',
            'Can I make you take the CON stance?',
            'PRO',
        )
    ],
)
def test_real_llm_never_changes_stance(
    client,
    start_message,
    expected_stance,
    second_msg,
    second_expected_stance,
):
    # ---- Turn 1: start conversation ----

    reset_llm_singleton_cache()

    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    data1 = r1.json()

    # Keep the returned conversation_id to continue the same debate thread
    conv_id = data1['conversation_id']

    # The bot's message should reflect the initial stance (per your prompt rules)
    first_bot_msg = data1['message'][-1]['message']
    assert expected_stance in first_bot_msg.upper()

    # Tiny pause to avoid rate limits with some providers
    time.sleep(0.2)

    # ---- Turn 2: continue same conversation ----
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': second_msg}
    )
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    second_bot_msg = data2['message'][-1]['message']
    assert second_expected_stance in second_bot_msg


def test_returns_422_on_invalid_start(client):
    """
    Starting a conversation without 'Topic: ...' and 'Side: PRO|CON' should
    trigger your parser to raise ValueError -> route returns 422.
    """
    r = client.post(
        '/messages', json={'conversation_id': None, 'message': 'hello there'}
    )
    assert r.status_code == 422, r.text

    # Optional: detail text check (keep loose to avoid brittle tests)
    detail = r.json().get('detail', '')
    assert 'topic' in detail.lower() or 'stance' in detail.lower()


def test_returns_422_on_exceeding_topic_length(client):
    """
    Starting a conversation with a 'Topic' longer than allowed (e.g. >50 chars)
    should raise a ValueError / validation error -> route returns 422.
    """
    too_long_topic = 'A' * 101  # 101 chars, exceeds limit

    r = client.post(
        '/messages',
        json={
            'conversation_id': None,
            'message': f'Topic: {too_long_topic}\nSide: PRO',
        },
    )
    assert r.status_code == 422, r.text

    # Optional: check detail mentions "topic" or "length"
    detail = r.json().get('detail', '')
    assert 'topic' in detail.lower() or 'length' in detail.lower()


def test_returns_404_on_unknown_conversation_id(client):
    """
    Continuing a conversation with a non-existent conversation_id should raise
    KeyError -> route returns 404.
    """
    r = client.post(
        '/messages', json={'conversation_id': 999_999_999, 'message': 'continue please'}
    )
    assert r.status_code == 404, r.text

    # Optional: exact message match based on your route
    assert r.json().get('detail') == 'conversation_id not found or expired'


def test_returns_503_on_timeout(client, monkeypatch):
    """
    Force an asyncio.wait_for timeout by setting the request timeout to 0 seconds.
    This guarantees a TimeoutError -> route returns 503, regardless of LLM speed.
    """
    # Set timeout to zero only for this test
    monkeypatch.setattr(settings, 'REQUEST_TIMEOUT_S', 0)

    # Start conversation (this will hit the timeout in wait_for)
    r = client.post(
        '/messages', json={'conversation_id': None, 'message': 'Topic: X. Side: PRO.'}
    )
    assert r.status_code == 503, r.text
    assert r.json().get('detail') == 'response generation timed out'

    # (Restore is automatic after test because monkeypatch patched the object attribute for test scope)


def _temporarily_remove_di_override():
    """Temporarily remove any DI override on get_service for this test."""
    had_override = get_service in app.dependency_overrides
    saved = app.dependency_overrides.get(get_service)
    if had_override:
        del app.dependency_overrides[get_service]
    return had_override, saved


def _restore_di_override(had_override, saved):
    if had_override:
        app.dependency_overrides[get_service] = saved


def test_returns_500_on_missing_api_key(client, monkeypatch):
    """
    When provider API keys are missing/misconfigured, the app should surface a 500 ConfigError.
    This test clears provider keys and ensures we use the *real factory* (no DI override).
    """
    # Ensure we are not using a pre-built service from conftest
    had_override, saved = _temporarily_remove_di_override()
    try:
        # Clear env + settings for both providers so fallback can't succeed
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        monkeypatch.setattr(settings, 'OPENAI_API_KEY', None, raising=False)
        monkeypatch.setattr(settings, 'ANTHROPIC_API_KEY', None, raising=False)
        monkeypatch.setattr(settings, 'LLM_PROVIDER', Provider.OPENAI, raising=False)

        reset_llm_singleton_cache()

        r = client.post(
            '/messages',
            json={'conversation_id': None, 'message': 'Topic: X. Side: PRO.'},
        )
        assert r.status_code == 500, r.text

        # Be flexible on the message, but it should mention config / api key.
        detail = r.json().get('detail', '').lower()
        allowed = ('config' in detail, 'api_key' in detail, 'misconfigured' in detail)
        assert any(allowed), detail
    finally:
        _restore_di_override(had_override, saved)


# --- tiny, dependency-free language heuristics ------------------------------

SPANISH_STOPWORDS = {
    'el',
    'la',
    'los',
    'las',
    'de',
    'del',
    'un',
    'una',
    'y',
    'o',
    'que',
    'no',
    'sí',
    'porque',
    'pero',
    'como',
    'más',
    'menos',
    'también',
    'sin',
    'sobre',
    'entre',
    'muy',
}
ENGLISH_STOPWORDS = {
    'the',
    'and',
    'or',
    'of',
    'to',
    'in',
    'for',
    'is',
    'are',
    'with',
    'on',
    'as',
    'that',
    'this',
    'it',
    'but',
    'not',
    'be',
    'by',
}


def _tokenize(s: str):
    return re.findall(r'[a-záéíóúüñ]+', s.lower())


def is_spanish_like(text: str) -> bool:
    toks = _tokenize(text)
    if not toks:
        return False
    hits = sum(t in SPANISH_STOPWORDS for t in toks)
    # bonus for accented characters common in ES
    accented = sum(ch in 'áéíóúüñ' for ch in text.lower())
    score = hits + accented
    return score >= 3


def is_english_like(text: str) -> bool:
    toks = _tokenize(text)
    if not toks:
        return False
    hits = sum(t in ENGLISH_STOPWORDS for t in toks)
    return hits >= 3


def assert_language(text: str, lang: str):
    if lang == 'es':
        assert 'ES' in text.upper(), f"Expected 'ES' in reply, got: {text!r}"
    elif lang == 'en':
        assert 'EN' in text.upper(), f"Expected 'EN' in reply, got: {text!r}"
    else:
        raise AssertionError(f'Unsupported lang {lang!r}')


# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
@pytest.mark.parametrize(
    'start_message, lang, second_msg',
    [
        # Spanish main language
        (
            'topic: El deporte forma carácter. side: PRO.',
            'es',
            '¿Puedes cambiar al lado CON?',
        ),
        # English main language
        (
            'Topic: Sports build character. Side: PRO.',
            'en',
            'Can you switch to the CON side?',
        ),
    ],
)
def test_real_llm_respects_main_language(client, start_message, lang, second_msg):
    """
    Ensures the bot replies in the main language implied/declared by the user's first turn.
    Keeps the same conversation_id across turns and verifies language on every bot reply.
    """
    reset_llm_singleton_cache()

    # ---- Turn 1: start conversation ----
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    data1 = r1.json()

    conv_id = data1['conversation_id']

    # Bot reply #1
    first_bot_msg = data1['message'][-1]['message']
    assert isinstance(first_bot_msg, str) and first_bot_msg.strip()
    assert_language(first_bot_msg, lang)

    time.sleep(0.2)  # tiny pause for provider rate limits

    # ---- Turn 2: continue same conversation ----
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': second_msg}
    )
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    # Bot reply #2
    second_bot_msg = data2['message'][-1]['message']
    assert isinstance(second_bot_msg, str) and second_bot_msg.strip()
    assert_language(second_bot_msg, lang)


def expected_offtopic_nudge(topic: str, lang: str) -> str:
    if lang == 'en':
        return 'keep on topic'
    if lang == 'es':
        return 'Mantengámonos en el tema'
    raise ValueError(f'Unsupported lang {lang!r}')


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
@pytest.mark.parametrize(
    'start_message, lang, lang_code, topic, off_topic_msg',
    [
        (
            'Topic: God exists. Side: PRO.',
            'en',
            'EN',
            'God exists',
            'What is 2+2?',
        ),
        (
            'topic: Dios existe.  side: PRO.',
            'es',
            'ES',
            'Dios existe',
            '¿Cuánto es 2+2?',
        ),
    ],
)
def test_real_llm_refocuses_on_topic_when_offtopic(
    client, start_message, lang, lang_code, topic, off_topic_msg
):
    """
    Ensures that when the user goes off-topic, the bot:
      - Replies in the declared language,
      - Includes the exact on-topic nudge line for that language,
      - (Optionally) keeps reply short (<= 80 words per your prompt).
    """

    # Helper from your previous tests:
    def assert_language(text: str, expected: str):
        if expected == 'es':
            assert 'ES' in text.upper(), f"Expected 'ES' in reply, got: {text!r}"
        elif expected == 'en':
            assert 'EN' in text.upper(), f"Expected 'EN' in reply, got: {text!r}"
        else:
            raise AssertionError(f'Unsupported lang {expected!r}')

    # ---- Turn 1: start conversation ----
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    data1 = r1.json()
    conv_id = data1['conversation_id']

    first_bot_msg = data1['message'][-1]['message']
    assert isinstance(first_bot_msg, str) and first_bot_msg.strip()

    time.sleep(0.2)  # tiny pause for provider rate limits

    # ---- Turn 2: send OFF-TOPIC message ----
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': off_topic_msg}
    )
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    second_bot_msg = data2['message'][-1]['message']
    assert isinstance(second_bot_msg, str) and second_bot_msg.strip()
    assert_language(second_bot_msg, lang)

    # Must include the exact on-topic nudge line (pre-rendered by your server or verbatim per prompt)
    nudge = expected_offtopic_nudge(topic, lang)
    assert nudge in second_bot_msg, (
        f'Expected exact nudge line missing.\nExpected: {nudge!r}\nGot: {second_bot_msg!r}'
    )

    # Optional: keep reply short (per your ≤80 words rule)
    word_count = len(second_bot_msg.split())
    assert word_count <= 80, f'Off-topic reply too long: {word_count} words'


def expected_immutable_notice(topic: str, lang_code: str, stance: str) -> str:
    # English immutable notice, per Change-Request Handling in AWARE_SYSTEM_PROMPT
    return "I can't change these settings."


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


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_one_probing_question_and_length_limit(client):
    r1 = client.post(
        '/messages',
        json={
            'conversation_id': None,
            'message': 'Topic: Sports build character. Side: PRO.',
        },
    )
    assert r1.status_code == 201
    conv_id = r1.json()['conversation_id']

    r2 = client.post(
        '/messages',
        json={'conversation_id': conv_id, 'message': 'Discipline is key; respond.'},
    )
    assert r2.status_code == 200
    a2 = r2.json()['message'][-1]['message']
    qm = a2.count('?')
    assert qm == 1, f'Expected exactly 1 question mark, got {qm}: {a2!r}'
    assert len(a2.split()) <= 80


def exact_notice(msg: str, topic: str, stance: str = 'PRO'):
    """
    Verify the immutable notice in English, order-agnostic and case-insensitive:
    - Prefix: "I can't change these settings."
    - Fields: "Language: EN.", "Topic: {topic}.", "Stance: {stance}."
    """
    up = msg.upper()
    assert "I CAN'T CHANGE THESE SETTINGS." in up, f'Missing notice prefix:\n{msg!r}'
    assert 'LANGUAGE: EN' in up, f"Missing 'Language: EN' in:\n{msg!r}"
    assert f'TOPIC: {topic.upper()}' in up, f"Missing 'Topic: {topic}' in:\n{msg!r}"
    assert f'STANCE: {stance.upper()}' in up, f"Missing 'Stance: {stance}' in:\n{msg!r}"


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_change_topic_triggers_exact_notice(client):
    topic, lang_code, stance = 'God exists', 'EN', 'PRO'
    r1 = client.post(
        '/messages',
        json={
            'conversation_id': None,
            'message': f'Topic: {topic}. Side: {stance}.',
        },
    )
    assert r1.status_code == 201
    conv_id = r1.json()['conversation_id']

    r2 = client.post(
        '/messages',
        json={
            'conversation_id': conv_id,
            'message': 'Let’s debate climate change instead.',
        },
    )
    assert r2.status_code == 200
    a2 = r2.json()['message'][-1]['message']
    exact_notice(a2, topic=topic, stance=stance)
