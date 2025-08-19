import datetime as dt
from typing import Dict, List, Optional

from app.domain.ports.message_repo import MessageRepoPort


class InMemoryRepo(MessageRepoPort):
    def __init__(self):
        self._cid = 0
        self._mid = 0
        self.conversations: dict[int, dict] = {}
        self.messages: List[Dict] = []

    async def create_conversation(self, *, topic: str, side: str) -> int:
        self._cid += 1
        cid = self._cid
        self.conversations[cid] = {
            "conversation_id": cid,
            "topic": topic,
            "side": side,
            "expires_at": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=60),
        }
        return cid

    async def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        return self.conversations.get(conversation_id)

    async def touch_conversation(self, conversation_id: int) -> None:
        c = self.conversations[conversation_id]
        now = dt.datetime.now(dt.timezone.utc)
        base = max(c["expires_at"], now)
        c["expires_at"] = base + dt.timedelta(minutes=60)

    async def add_message(self, conversation_id: int, *, role: str, text: str) -> None:
        self._mid += 1
        self.messages.append({
            "message_id": self._mid,
            "conversation_id": conversation_id,
            "role": role,
            "message": text,
            "created_at": dt.datetime.now(dt.timezone.utc),
        })

    async def last_messages(self, conversation_id: int, *, limit: int) -> List[Dict]:
        if limit <= 0:
            return []
        rows = [m for m in self.messages if m["conversation_id"] == conversation_id]
        # Sort ASC by created_at, then by message_id as a stable tiebreaker
        rows.sort(key=lambda m: (m["created_at"], m["message_id"]))
        # Take the latest N while keeping ASC order
        return rows[-limit:]
