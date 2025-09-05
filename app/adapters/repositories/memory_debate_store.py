# adapters/repositories/in_memory_debate_store.py
from copy import deepcopy
from typing import Callable, Optional

from app.domain.concession_policy import DebateState  # adjust import to your tree
from app.domain.ports.debate_store import DebateStorePort


class InMemoryDebateStore(DebateStorePort):
    """
    Minimal in-memory DebateStore keyed by conversation_id.
    DebateState does NOT need an `id` field. Not thread-safe.
    """

    def __init__(self):
        self._db: dict[int, DebateState] = {}

    def get(self, conversation_id: int) -> Optional[DebateState]:
        s = self._db.get(conversation_id)
        return deepcopy(s) if s else None

    def create(
        self, conversation_id: int, *, stance: str, lang: str = 'es'
    ) -> DebateState:
        if conversation_id in self._db:
            raise ValueError(f'DebateState {conversation_id} already exists')
        s = DebateState(stance=stance, lang=lang)
        self._db[conversation_id] = deepcopy(s)
        return deepcopy(s)

    def save(self, conversation_id: int, state: DebateState) -> None:
        self._db[conversation_id] = deepcopy(state)

    def update(
        self, conversation_id: int, fn: Callable[[DebateState], None]
    ) -> DebateState:
        s = self._db.get(conversation_id)
        if s is None:
            raise KeyError(f'DebateState not found: {conversation_id}')
        fn(s)  # mutate in place
        self._db[conversation_id] = s
        return deepcopy(s)

    # Optional helpers
    def exists(self, conversation_id: int) -> bool:
        return conversation_id in self._db

    def clear(self) -> None:
        self._db.clear()
        self._db.clear()
