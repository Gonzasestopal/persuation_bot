from typing import Callable, Optional, Protocol

from app.domain.concession_policy import DebateState


class DebateStorePort(Protocol):
    async def get(self, conversation_id: int) -> Optional[DebateState]:
        pass

    async def get_or_create(
        self,
        conversation_id: int,
        stance: str,
        lang: str,
    ) -> DebateState:
        pass

    async def save(self, state: DebateState) -> None:
        pass

    async def update(
        self, conversation_id: int, fn: Callable[[DebateState], None]
    ) -> DebateState:
        pass
