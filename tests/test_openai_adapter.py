from datetime import datetime
from unittest.mock import Mock

import pytest

from app.adapters.llm.constants import OpenAIModels
from app.adapters.llm.openai import OpenAIAdapter
from app.domain.models import Conversation, Message


def test_adapter_config():
    client = Mock()
    adapter = OpenAIAdapter(
        max_history=5,
        api_key='test',
        client=client,
    )

    assert adapter.client == client
    assert adapter.temperature == 0.3
    assert adapter.max_history == 5


@pytest.mark.asyncio
async def test_adapter_generate():
    client = Mock()
    adapter = OpenAIAdapter(
        max_history=5,
        api_key='test',
        client=client,
    )

    expires_at = datetime.utcnow()
    conversation = Conversation(id=1, topic='X', side='con', expires_at=expires_at)

    with pytest.raises(NotImplementedError):
        await adapter.generate(conversation=conversation)


@pytest.mark.asyncio
async def test_adapter_debate():
    client = Mock()
    adapter = OpenAIAdapter(
        max_history=5,
        api_key='test',
        client=client,
    )

    user_topic = 'Dogs are human best friend because they are loyal'
    bot_reply = 'Not really, they are not loyal, its their instict to survive.'

    messages = [
        Message(role='user', message=user_topic),
        Message(role='bot', message=bot_reply)
    ]

    with pytest.raises(NotImplementedError):
        await adapter.debate(messages=messages)


@pytest.mark.asyncio
async def test_debate_raises_if_exceeds_history_limit():
    client = Mock()
    adapter = OpenAIAdapter(api_key="sk-test", max_history=2, client=client)

    messages = [
        Message(role="user", message="u1"),
        Message(role="bot", message="b1"),
        Message(role="user", message="u2"),  # 3 messages > limit=2
    ]

    with pytest.raises(ValueError) as e:
        await adapter.debate(messages=messages)

    assert "exceeds history limit" in str(e.value).lower()


def test_openai_adapter_accepts_openai_models():
    a = OpenAIAdapter(api_key="sk-test", max_history=10, model=OpenAIModels.GPT_4O)
    assert a.model == "gpt-4o"


def test_openai_adapter_accepts_string_model():
    a = OpenAIAdapter(api_key="sk-test", max_history=10, model="gpt-4o")
    assert a.model == "gpt-4o"


def test_openai_adapter_rejects_anthropic_model():
    with pytest.raises(ValueError) as e:
        OpenAIAdapter(api_key="sk-test", max_history=10, model='claude-5')
    assert "not a valid openai model" in str(e.value).lower()
