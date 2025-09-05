from functools import lru_cache

from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
from app.domain.ports.debate_store import DebateStorePort


@lru_cache(maxsize=1)
def get_state_store() -> DebateStorePort:
    # single, per-process store
    return InMemoryDebateStore()
