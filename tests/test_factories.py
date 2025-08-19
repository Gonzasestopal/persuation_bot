from types import SimpleNamespace

import pytest

import app.factories as fx
from app.adapters.llm.constants import OpenAIModels, Provider
from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.openai import OpenAIAdapter
from app.factories import get_llm


def stub_settings(monkeypatch, **vals):
    st = SimpleNamespace(
        LLM_PROVIDER=Provider.OPENAI,
        LLM_MODEL=OpenAIModels.GPT_4O.value,
        OPENAI_API_KEY="sk-test",   # default
        HISTORY_LIMIT=10,
        **vals,
    )
    monkeypatch.setattr(fx, "settings", st)


def test_llm_factory_openai():
    provider = 'openai'
    llm = get_llm(provider=provider)
    assert isinstance(llm, OpenAIAdapter)


def test_llm_factory_default():
    llm = get_llm()
    assert isinstance(llm, DummyLLMAdapter)


def test_openai_adapter_accepts_openai_models():
    a = get_llm(provider=Provider.OPENAI, model=OpenAIModels.GPT_4O)
    assert a.model == "gpt-4o"


def test_openai_adapter_accepts_string_model():
    a = get_llm(provider=Provider.OPENAI, model="gpt-4o")
    assert a.model == "gpt-4o"


def test_openai_adapter_rejects_anthropic_model():
    with pytest.raises(ValueError) as e:
        get_llm(provider=Provider.OPENAI, model="claude-5")
    assert "not a valid openai model" in str(e.value).lower()


def test_openai_empty_api_key_raises(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="")

    with pytest.raises(ValueError) as e:
        fx.get_llm(provider="openai")

    assert "OPENAI_API_KEY is required" in str(e.value)
