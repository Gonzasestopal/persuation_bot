from typing import Optional

from app.domain.parser import assert_no_topic_or_side_markers
from app.repositories.base import MessageRepoInterface


class MessageService(object):
    def __init__(self, parser, repo: MessageRepoInterface, history_limit=5):
        self.parser = parser
        self.repo = repo
        self.history_limit = history_limit

    async def handle(self, message: str, conversation_id: Optional[int] = None):
        if conversation_id is None:
            topic, side = self.parser(message)
            return await self.start_conversation(topic, side, message)

        assert_no_topic_or_side_markers(message)
        return await self.continue_conversation(message, conversation_id)

    async def start_conversation(self, topic: str, side: str, message: str = None):
        conversation_id = await self.repo.create_conversation(topic=topic, side=side)
        await self.repo.add_message(conversation_id=conversation_id, role="user", text=message)
        await self.repo.add_message(conversation_id=conversation_id, role="bot", text="bot reply")

        return {
            "conversation_id": conversation_id,
            "message": await self.repo.last_messages(
                conversation_id=conversation_id,
                limit=self.history_limit * 2,
            ),
        }

    async def continue_conversation(self, message: str, conversation_id: int):
        conversation = await self.repo.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise KeyError("conversation_id not found or expired")

        cid = conversation["conversation_id"]
        await self.repo.touch_conversation(conversation_id=cid)
        await self.repo.add_message(conversation_id=cid, role="user", text=message)
        await self.repo.add_message(conversation_id=cid, role="bot", text="bot reply")

        return {
            "conversation_id": cid,
            "message": await self.repo.last_messages(conversation_id=cid, limit=self.history_limit * 2),
        }
