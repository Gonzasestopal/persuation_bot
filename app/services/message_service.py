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
            return await self.start_conversation(topic, side)

        assert_no_topic_or_side_markers(message)
        return await self.continue_conversation(message, conversation_id)

    async def start_conversation(self, topic: str, side: str):
        raise NotImplementedError

    async def continue_conversation(self, message: str, conversation_id: int):
        raise NotImplementedError
