from unittest.mock import AsyncMock

import pytest

from app.llm.interface import LLMAdapterInterface


class DummyLLM(LLMAdapterInterface):
    async def generate(self, messages):
        # simple echo for testing
        return "bot reply"


@pytest.mark.asyncio
async def test_llm_interface_can_generate():
    llm = DummyLLM()
    reply = await llm.generate([{"role": "user", "message": "Hello"}])
    assert reply == "bot reply"


@pytest.mark.asyncio
async def test_llm_adapter_is_mockable():
    llm = AsyncMock(spec=LLMAdapterInterface)
    llm.generate.return_value = "bot reply"

    reply = await llm.generate([{"role": "user", "message": "Hi"}])

    llm.generate.assert_awaited_once_with([{"role": "user", "message": "Hi"}])
    assert reply == "bot reply"
