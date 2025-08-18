from datetime import datetime, timezone
from typing import Optional

from app.domain.parser import assert_no_topic_or_side_markers
from app.domain.ports.llm import LLMPort
from app.repositories.base import MessageRepoInterface


class MessageService(object):
    def __init__(
            self,
            parser,
            repo: MessageRepoInterface,
            llm: LLMPort,
            history_limit=5,
        ):
        self.parser = parser
        self.repo = repo
        self.history_limit = history_limit
        self.llm = llm

    async def handle(self, message: str, conversation_id: Optional[int] = None):
        if conversation_id is None:
            topic, side = self.parser(message)
            return await self.start_conversation(topic, side, message)

        assert_no_topic_or_side_markers(message)
        return await self.continue_conversation(message, conversation_id)

    async def start_conversation(self, topic: str, side: str, message: str = None):
        conversation_id = await self.repo.create_conversation(topic=topic, side=side)

        # store user message
        await self.repo.add_message(conversation_id=conversation_id, role="user", text=message)

        # fetch recent history and call LLM
        history = await self.repo.last_messages(conversation_id=conversation_id, limit=self.history_limit * 2)
        reply = await self.llm.generate(history)

        # store LLM reply
        await self.repo.add_message(conversation_id=conversation_id, role="bot", text=reply)

        return {
            "conversation_id": conversation_id,
            "message": await self.repo.last_messages(conversation_id=conversation_id, limit=self.history_limit * 2),
        }

    async def continue_conversation(self, message: str, conversation_id: int):
        conversation = await self.repo.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise KeyError("conversation_id not found")

        if conversation['expires_at'] <= datetime.now(timezone.utc):
            raise KeyError("conversation_id expired")

        cid = conversation["conversation_id"]
        await self.repo.touch_conversation(conversation_id=cid)
        await self.repo.add_message(conversation_id=cid, role="user", text=message)

        history = await self.repo.last_messages(conversation_id=cid, limit=self.history_limit * 2)
        reply = await self.llm.generate(history)
        await self.repo.add_message(conversation_id=cid, role="bot", text=reply)

        return {
            "conversation_id": cid,
            "message": await self.repo.last_messages(conversation_id=cid, limit=self.history_limit * 2),
        }
