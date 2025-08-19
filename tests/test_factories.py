from types import SimpleNamespace

import pytest

import app.factories as fx
from app.adapters.llm.constants import OpenAIModels, Provider
from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.openai import OpenAIAdapter
from app.domain.exceptions import ConfigError


def stub_settings(monkeypatch, **overrides):
    """Replace app.factories.settings with a fake namespace."""
    defaults = dict(
        OPENAI_API_KEY="test-key",
        LLM_PROVIDER="openai",
        HISTORY_LIMIT=5,
        REQUEST_TIMEOUT_S=30,
    )

    # let overrides replace defaults cleanly
    merged = {**defaults, **overrides}

    monkeypatch.setattr("app.factories.settings", SimpleNamespace(**merged))


def test_llm_factory_openai(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="sk-test")
    llm = fx.get_llm(provider="openai", model="gpt-4o")
    assert isinstance(llm, OpenAIAdapter)


def test_llm_factory_default():
    llm = fx.get_llm()
    assert isinstance(llm, DummyLLMAdapter)


def test_openai_adapter_accepts_openai_models(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="sk-test")
    a = fx.get_llm(provider=Provider.OPENAI.value, model=OpenAIModels.GPT_4O)
    assert isinstance(a, OpenAIAdapter)
    assert a.model == "gpt-4o"


def test_openai_adapter_accepts_openai_models_string(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="sk-test")
    a = fx.get_llm(provider=Provider.OPENAI.value, model="gpt-4o")
    assert isinstance(a, OpenAIAdapter)
    assert a.model == "gpt-4o"


def test_openai_adapter_rejects_anthropic_model(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="sk-test")

    with pytest.raises(ConfigError) as e:
        fx.get_llm(provider=Provider.OPENAI, model="claude-5")
    assert "not a valid openai model" in str(e.value).lower()


def test_openai_empty_api_key_raises(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="")

    with pytest.raises(ConfigError) as e:
        fx.get_llm(provider="openai")

    assert "OPENAI_API_KEY is required" in str(e.value)
