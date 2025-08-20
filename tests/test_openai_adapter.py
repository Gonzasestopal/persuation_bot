from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.adapters.llm.openai import OpenAIAdapter
from app.domain.models import Conversation, Message


class FakeResponses:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="FAKE-OUTPUT")


class FakeClient:
    def __init__(self, calls):
        self.responses = FakeResponses(calls)


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
async def test_adapter_generate_builds_prompt_and_returns_output(monkeypatch):
    calls = []
    client = FakeClient(calls)
    expires_at = datetime.utcnow()
    adapter = OpenAIAdapter(max_history=5, api_key="sk-test", client=client, model="gpt-4o", temperature=0.3)

    conv = Conversation(id=1, topic="X", side="con", expires_at=expires_at)
    out = await adapter.generate(conversation=conv)

    assert out == "FAKE-OUTPUT"
    assert len(calls) == 1
    sent = calls[0]

    assert sent["model"] == "gpt-4o"
    assert sent["temperature"] == 0.3

    msgs = sent["input"]
    assert msgs[0] == {"role": "system", "content": adapter.system_prompt}
    assert msgs[1]["role"] == "user"
    assert "You are debating the topic 'X'" in msgs[1]["content"]
    assert "Take the con side." in msgs[1]["content"]


@pytest.mark.asyncio
async def test_adapter_debate_maps_roles_and_respects_history(monkeypatch):
    calls = []
    client = FakeClient(calls)
    adapter = OpenAIAdapter(max_history=10, api_key="sk-test", client=client, model="gpt-4o", temperature=0.2)

    msgs = [
        Message(role="user", message="u1"),
        Message(role="bot",  message="b1"),
        Message(role="user", message="u2"),
        Message(role="bot",  message="b2"),
    ]

    out = await adapter.debate(messages=msgs)
    assert out == "FAKE-OUTPUT"

    assert len(calls) == 1
    sent = calls[0]
    assert sent["model"] == "gpt-4o"
    assert sent["temperature"] == 0.2

    input_msgs = sent["input"]
    assert input_msgs[0] == {"role": "system", "content": adapter.system_prompt}

    role_map = {"user": "user", "bot": "assistant"}
    assert len(input_msgs) == len(msgs) + 1

    for i, m in enumerate(msgs, start=1):
        assert set(input_msgs[i].keys()) == {"role", "content"}
        assert input_msgs[i]["role"] == role_map[m.role]
        assert input_msgs[i]["content"] == m.message


@pytest.mark.asyncio
async def test_adapter_debate_raises_when_over_history():
    adapter = OpenAIAdapter(max_history=2, api_key="sk", client=FakeClient([]))
    msgs = [Message(role="user", message="u1"),
            Message(role="bot",  message="b1"),
            Message(role="user", message="u2")]

    with pytest.raises(ValueError) as e:
        await adapter.debate(messages=msgs)

    assert "exceeds history limit" in str(e.value).lower()
