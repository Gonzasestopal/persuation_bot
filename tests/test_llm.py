from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from app.adapters.llm.dummy import DummyLLMAdapter
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


@pytest.mark.asyncio
async def test_llm_interface_can_generate():
    expires_at = datetime.utcnow()
    conv = Conversation(id=1, topic='X', side='con', expires_at=expires_at)
    llm = DummyLLMAdapter()
    reply = await llm.generate(conversation=conv)
    assert reply == f"I am a bot that defends {conv.topic} and im side {conv.side}"


@pytest.mark.asyncio
async def test_llm_adapter_is_mockable():
    expires_at = datetime.utcnow()
    conv = Conversation(id=1, topic='X', side='con', expires_at=expires_at)
    llm = AsyncMock(spec=LLMPort)
    llm.generate.return_value = "bot reply"

    reply = await llm.generate(conversation=conv)

    llm.generate.assert_awaited_once_with(conversation=conv)
    assert reply == "bot reply"


@pytest.mark.asyncio
async def test_llm_interface_can_debate():
    user_topic = 'Dogs are human best friend because they are loyal'
    bot_reply = 'Not really, they are not loyal, its their instict to survive.'
    user_message = Message(role='user', message=user_topic)
    bot_message = Message(role='bot', message=bot_reply)
    llm = DummyLLMAdapter()
    reply = await llm.debate(messages=[user_message, bot_message])
    assert "After considering your last 2 messages" in reply
