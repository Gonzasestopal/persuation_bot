import os
import time

import pytest

from app.infra.llm import reset_llm_singleton_cache

pytestmark = pytest.mark.integration


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
