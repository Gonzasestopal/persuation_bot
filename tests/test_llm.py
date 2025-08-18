from unittest.mock import AsyncMock

import pytest

from app.adapters.llm.dummy import DummyLLMAdapter
from app.domain.ports.llm import LLMPort


@pytest.mark.asyncio
async def test_llm_interface_can_generate():
    llm = DummyLLMAdapter()
    reply = await llm.generate([{"role": "user", "message": "Hello"}])
    assert reply == "bot reply"


@pytest.mark.asyncio
async def test_llm_adapter_is_mockable():
    llm = AsyncMock(spec=LLMPort)
    llm.generate.return_value = "bot reply"

    reply = await llm.generate([{"role": "user", "message": "Hi"}])

    llm.generate.assert_awaited_once_with([{"role": "user", "message": "Hi"}])
    assert reply == "bot reply"
