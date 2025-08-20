from typing import List, Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.domain.models import Conversation, Message
from app.domain.ports.message_repo import MessageRepoPort


class PgMessageRepo(MessageRepoPort):
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_conversation(self, *, topic: str, side: str) -> Conversation:
        q = "INSERT INTO conversations (topic, side) VALUES (%s, %s) RETURNING conversation_id, expires_at"
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(q, (topic, side))
            (cid, expires_at) = await cur.fetchone()
            return Conversation(id=cid, topic=topic, side=side, expires_at=expires_at)

    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        q = "SELECT conversation_id AS id, topic, side, expires_at FROM conversations WHERE conversation_id = %s"
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(q, (conversation_id,))
            row = await cur.fetchone()
            return Conversation(**row) if row else None

    async def touch_conversation(self, conversation_id: int) -> None:
        q = """UPDATE conversations
               SET expires_at = GREATEST(expires_at, NOW()) + INTERVAL '60 minutes'
               WHERE conversation_id = %s"""
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(q, (conversation_id,))

    async def add_message(self, conversation_id: int, *, role: str, text: str) -> None:
        q = "INSERT INTO messages (conversation_id, role, message) VALUES (%s, %s, %s) RETURNING message_id"
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(q, (conversation_id, role, text))

    async def last_messages(self, conversation_id: int, *, limit: int) -> List[Message]:
        q = """
        SELECT role, message, created_at
        FROM (
            SELECT role, message, created_at, message_id
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at DESC, message_id DESC
            LIMIT %s
        ) sub
        ORDER BY created_at ASC, message_id ASC
        """
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(q, (conversation_id, limit))
            rows = await cur.fetchall()
            return [Message(**dict(r)) for r in rows]

    async def all_messages(self, conversation_id: int) -> List[Message]:
        q = """
        SELECT role, message, created_at
        FROM messages
        WHERE conversation_id = %s
        """
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(q, (conversation_id,))
            rows = await cur.fetchall()
            return [Message(**dict(r)) for r in rows]
