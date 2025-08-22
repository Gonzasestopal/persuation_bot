# app/repositories/inmemory_message_repo.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from app.domain.models import Conversation, Message
from app.domain.ports.message_repo import MessageRepoPort


def _now_utc() -> datetime:
    # Always timezone-aware (UTC)
    return datetime.now(timezone.utc)


def _to_aware_utc(dt: datetime) -> datetime:
    # Normalize any incoming naive/aware datetime to aware UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class InMemoryMessageRepo(MessageRepoPort):
    def __init__(self, ttl_minutes: int = 60) -> None:
        self._convs: Dict[int, Conversation] = {}
        self._msgs: Dict[int, List[Message]] = {}
        self._next_id = 1
        self._ttl = timedelta(minutes=ttl_minutes)

    async def create_conversation(self, *, topic: str, side: str) -> Conversation:
        cid = self._next_id
        self._next_id += 1
        expires_at = _now_utc() + self._ttl
        conv = Conversation(id=cid, topic=topic, side=side, expires_at=expires_at)
        self._convs[cid] = conv
        self._msgs[cid] = []
        return conv

    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        conv = self._convs.get(conversation_id)
        if conv and conv.expires_at:
            conv.expires_at = _to_aware_utc(conv.expires_at)
        return conv

    async def touch_conversation(self, conversation_id: int) -> None:
        conv = self._convs.get(conversation_id)
        if conv:
            now = _now_utc()
            current_exp = _to_aware_utc(conv.expires_at) if conv.expires_at else now
            conv.expires_at = max(current_exp, now) + self._ttl

    async def add_message(self, conversation_id: int, *, role: str, text: str) -> None:
        self._msgs.setdefault(conversation_id, [])
        self._msgs[conversation_id].append(
            Message(role=role, message=text, created_at=_now_utc())
        )

    async def last_messages(self, conversation_id: int, *, limit: int) -> List[Message]:
        msgs = self._msgs.get(conversation_id, [])
        # `created_at` is already aware UTC; keep insertion order oldest→newest
        return list(msgs[-limit:])

    async def all_messages(self, conversation_id: int) -> List[Message]:
        return list(self._msgs.get(conversation_id, []))
