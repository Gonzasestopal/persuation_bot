from functools import lru_cache

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.ports.nli import NLIPort


@lru_cache(maxsize=1)
def get_nli_singleton() -> NLIPort:
    # single, per-process store
    return HFNLIProvider()
