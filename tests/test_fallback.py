# tests/test_fallback_llm_unit.py
import asyncio
from datetime import datetime

import pytest

from app.adapters.llm.fallback import FallbackLLM
from app.domain import exceptions as de
from app.domain.models import Conversation, Message


class OKLLM:
    async def generate(self, conversation): return "ok-gen"
    async def debate(self, messages): return "ok-debate"


class TimeoutLLM:
    async def generate(self, conversation): await asyncio.sleep(1e9)
    async def debate(self, messages): await asyncio.sleep(1e9)


class FailLLM:
    async def generate(self, conversation): raise RuntimeError("boom")
    async def debate(self, messages): raise RuntimeError("boom")

@pytest.mark.asyncio
async def test_sequential_primary_ok():
    expires_at = datetime.utcnow()
    fb = FallbackLLM(primary=OKLLM(), secondary=FailLLM(), per_provider_timeout_s=0.1)
    out = await fb.generate(Conversation(id=1, topic="t", side="pro", expires_at=expires_at))
    assert out == "ok-gen"

@pytest.mark.asyncio
async def test_sequential_fallback_ok():
    fb = FallbackLLM(primary=FailLLM(), secondary=OKLLM(), per_provider_timeout_s=0.1)
    out = await fb.debate([Message(role="user", message="hi")])
    assert out == "ok-debate"

@pytest.mark.asyncio
async def test_both_fail_service_error():
    expires_at = datetime.utcnow()
    fb = FallbackLLM(primary=FailLLM(), secondary=FailLLM(), per_provider_timeout_s=0.1)
    with pytest.raises(de.LLMServiceError):
        await fb.generate(Conversation(id=1, topic="t", side="pro", expires_at=expires_at))

@pytest.mark.asyncio
async def test_both_timeout_llm_timeout():
    fb = FallbackLLM(primary=TimeoutLLM(), secondary=TimeoutLLM(), per_provider_timeout_s=0.01)
    with pytest.raises(de.LLMTimeout):
        await fb.debate([Message(role="user", message="hi")])
    fb = FallbackLLM(primary=TimeoutLLM(), secondary=TimeoutLLM(), per_provider_timeout_s=0.01)
    with pytest.raises(de.LLMTimeout):
        await fb.debate([Message(role="user", message="hi")])
