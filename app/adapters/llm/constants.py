from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"


class OpenAIModels(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
