from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.openai import OpenAIAdapter
from app.factories import get_llm


def test_llm_factory_openai():
    provider = 'openai'
    llm = get_llm(provider=provider)
    assert isinstance(llm, OpenAIAdapter)


def test_llm_factory_default():
    llm = get_llm()
    assert isinstance(llm, DummyLLMAdapter)

