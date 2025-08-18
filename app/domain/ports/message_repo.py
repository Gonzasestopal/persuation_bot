from typing import Dict, List, Optional, Protocol


class MessageRepoPort(Protocol):
    async def create_conversation(self, *, topic: str, side: str) -> int:
        raise NotImplementedError

    async def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        raise NotImplementedError

    async def touch_conversation(self, conversation_id: int) -> None:
        raise NotImplementedError

    async def add_message(self, conversation_id: int, *, role: str, text: str) -> None:
        raise NotImplementedError

    async def last_messages(self, conversation_id: int, *, limit: int) -> List[Dict]:
        raise NotImplementedError
