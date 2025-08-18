from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.repositories.base import MessageRepoInterface


class PgMessageRepo(MessageRepoInterface):
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_conversation(self, *, topic: str, side: str) -> int:
        q = "INSERT INTO conversations (topic, side) VALUES (%s, %s) RETURNING conversation_id"
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(q, (topic, side))
            (cid,) = await cur.fetchone()
            return cid

    async def get_conversation(self, conversation_id: int):
        q = "SELECT conversation_id, topic, side, expires_at FROM conversations WHERE conversation_id = %s"
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(q, (conversation_id,))
            row = await cur.fetchone()
            return dict(row) if row else None

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

    async def last_messages(self, conversation_id: int, *, limit: int):
        q = """SELECT role, message, created_at
               FROM messages
               WHERE conversation_id = %s
               ORDER BY created_at DESC
               LIMIT %s"""
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(q, (conversation_id, limit))
            rows = await cur.fetchall()
            rows.reverse()  # return oldest→newest
            return rows
