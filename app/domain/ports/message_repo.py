from typing import List, Optional, Protocol

from app.domain.models import Conversation, Message


class MessageRepoPort(Protocol):
    async def create_conversation(self, *, topic: str, side: str) -> int:
        raise NotImplementedError

    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        raise NotImplementedError

    async def touch_conversation(self, conversation_id: int) -> None:
        raise NotImplementedError

    async def add_message(self, conversation_id: int, *, role: str, text: str) -> None:
        raise NotImplementedError

    async def last_messages(self, conversation_id: int, *, limit: int) -> List[Message]:
        raise NotImplementedError

    async def all_messages(self, conversation_id: int) -> List[Message]:
        raise NotImplementedError
