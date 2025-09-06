from types import SimpleNamespace

import pytest

import app.infra.llm as fx
from app.adapters.llm.anthropic import AnthropicAdapter
from app.adapters.llm.constants import Difficulty, OpenAIModels, Provider
from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.openai import OpenAIAdapter
from app.domain.errors import ConfigError

pytestmark = pytest.mark.unit


def stub_settings(monkeypatch, **overrides):
    """
    Replace app.infra.llm.settings with a fake namespace.
    Only touches the llm wiring module, not global process env.
    """
    defaults = dict(
        OPENAI_API_KEY="test-key",
        LLM_PROVIDER="openai",
        ANTHROPIC_API_KEY="test-key",
        HISTORY_LIMIT=5,
        REQUEST_TIMEOUT_S=30,
        DIFFICULTY="easy",
        LLM_MODEL="gpt-4o",
        MAX_OUTPUT_TOKENS=120,
        LLM_PER_PROVIDER_TIMEOUT_S=25,
    )
    merged = {**defaults, **overrides}
    monkeypatch.setattr("app.infra.llm.settings", SimpleNamespace(**merged))


def test_llm_factory_openai(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="sk-test")
    llm = fx.get_llm(provider="openai", model="gpt-4o")
    assert isinstance(llm, OpenAIAdapter)


def test_llm_factory_default(monkeypatch):
    from enum import Enum

    class LLM(Enum):
        random = "random"

    stub_settings(monkeypatch, LLM_PROVIDER=LLM.random)
    # No provider → Dummy adapter by design
    llm = fx.get_llm()
    assert isinstance(llm, DummyLLMAdapter)


def test_openai_adapter_accepts_openai_models_enum_value(monkeypatch):
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
    # IMPORTANT: pass string or .value (the factory compares against str)
    with pytest.raises(ConfigError) as e:
        fx.get_llm(provider=Provider.OPENAI.value, model="claude-5")
    assert "not a valid openai model" in str(e.value).lower()


def test_openai_empty_api_key_raises(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="")
    with pytest.raises(ConfigError) as e:
        fx.get_llm(provider="openai")
    assert "openai_api_key is required" in str(e.value).lower()


def test_anthropic_api_key_raises(monkeypatch):
    stub_settings(monkeypatch, ANTHROPIC_API_KEY="")
    with pytest.raises(ConfigError) as e:
        fx.get_llm(provider="anthropic")
    assert "anthropic_api_key is required" in str(e.value).lower()


def test_set_debate_bot_difficulty_invalid(monkeypatch):
    stub_settings(monkeypatch, DIFFICULTY="hard")
    with pytest.raises(ConfigError) as e:
        fx.get_llm(provider="openai")
    assert "only easy and medium difficulty are supported" in str(e.value).lower()


def test_set_debate_bot_difficulty_empty(monkeypatch):
    stub_settings(monkeypatch, DIFFICULTY="")
    with pytest.raises(ConfigError) as e:
        fx.get_llm(provider="openai")
    assert "difficulty is required" in str(e.value).lower()


def test_set_debate_bot_difficulty_medium(monkeypatch):
    stub_settings(monkeypatch, DIFFICULTY="medium", OPENAI_API_KEY="sk-test")
    a = fx.get_llm(provider="openai", model="gpt-4o")
    assert a.difficulty == Difficulty.MEDIUM


def test_assert_is_openaiadapter(monkeypatch):
    stub_settings(monkeypatch, OPENAI_API_KEY="sk-test")
    llm = fx.make_openai()
    assert isinstance(llm, OpenAIAdapter)


def test_assert_is_anthropic(monkeypatch):
    stub_settings(monkeypatch, ANTHROPIC_API_KEY="sk-test")
    llm = fx.make_claude()
    assert isinstance(llm, AnthropicAdapter)
    assert isinstance(llm, AnthropicAdapter)
