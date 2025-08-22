from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.adapters.llm.anthropic import AnthropicAdapter
from app.domain.models import Conversation, Message


class FakeMessages:
    def __init__(self, calls):
        self.calls = calls

    async def create(self, **kwargs):
        # Capture the call
        self.calls.append(kwargs)
        # Return an Anthropic-like response: .content is a list of text blocks
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="FAKE-OUTPUT")]
        )


class FakeAsyncAnthropic:
    def __init__(self, calls):
        self.messages = FakeMessages(calls)


def test_adapter_config_defaults():
    calls = []
    client = FakeAsyncAnthropic(calls)

    adapter = AnthropicAdapter(
        api_key="test",
        client=client,  # ensure we use provided client
    )

    # Basic config checks
    assert adapter.client is client
    assert adapter.temperature == 0.3
    assert adapter.max_output_tokens == 120
    assert adapter.model in {"claude-3-5-sonnet-latest"}


@pytest.mark.asyncio
async def test_adapter_generate_builds_prompt_and_returns_output():
    calls = []
    client = FakeAsyncAnthropic(calls)

    adapter = AnthropicAdapter(
        api_key="sk-test",
        client=client,
        model="claude-3-5-sonnet-latest",
        temperature=0.3,
        max_output_tokens=120,
    )

    conv = Conversation(
        id=1,
        topic="X",
        side="con",
        expires_at=datetime.now(timezone.utc),
    )

    out = await adapter.generate(conversation=conv)
    assert out == "FAKE-OUTPUT"
    assert len(calls) == 1

    sent = calls[0]
    # Anthropic params
    assert sent["model"] == "claude-3-5-sonnet-latest"
    assert sent["temperature"] == 0.3
    assert sent["max_tokens"] == 120

    # System goes in top-level 'system'
    assert sent["system"] == adapter.system_prompt

    # Messages: only a single user turn for generate()
    msgs = sent["messages"]
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]["role"] == "user"
    # Anthropic content is a list of blocks
    content = msgs[0]["content"]
    assert isinstance(content, list) and content and content[0]["type"] == "text"

    text = content[0]["text"]
    assert "You are debating the topic 'X'" in text
    assert "Take the con side." in text


@pytest.mark.asyncio
async def test_adapter_debate_maps_roles_and_respects_history():
    calls = []
    client = FakeAsyncAnthropic(calls)

    adapter = AnthropicAdapter(
        api_key="sk-test",
        client=client,
        model="claude-3-5-sonnet-latest",
        temperature=0.2,
        max_output_tokens=90,
    )

    history = [
        Message(role="user", message="u1"),
        Message(role="bot",  message="b1"),
        Message(role="user", message="u2"),
        Message(role="bot",  message="b2"),
    ]

    out = await adapter.debate(messages=history)
    assert out == "FAKE-OUTPUT"
    assert len(calls) == 1

    sent = calls[0]
    assert sent["model"] == "claude-3-5-sonnet-latest"
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 90

    # System prompt is top-level
    assert sent["system"] == adapter.system_prompt

    # Verify mapping: 'user' -> 'user', 'bot' -> 'assistant'
    msgs = sent["messages"]
    assert len(msgs) == len(history)

    role_map = {"user": "user", "bot": "assistant"}
    for i, m in enumerate(history):
        sent_msg = msgs[i]
        assert set(sent_msg.keys()) == {"role", "content"}
        assert sent_msg["role"] == role_map[m.role]
        # Anthropic content is list of blocks
        assert isinstance(sent_msg["content"], list) and sent_msg["content"]
        block = sent_msg["content"][0]
        assert block["type"] == "text"
        assert block["text"] == m.message
        assert block["text"] == m.message
