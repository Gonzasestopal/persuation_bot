from dataclasses import dataclass


@dataclass(frozen=True)
class NLIConfig:
    model_name: str = 'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli'
    max_length: int = 512
    max_claims_per_turn: int = 3
