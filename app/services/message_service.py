from datetime import datetime, timezone
from typing import Optional

from app.domain.concession_policy import DebateState
from app.domain.errors import ConversationExpired, ConversationNotFound
from app.domain.parser import assert_no_topic_or_side_markers
from app.domain.ports.llm import LLMPort
from app.domain.ports.message_repo import MessageRepoPort
from app.services.concession_service import ConcessionService


class MessageService(object):
    def __init__(
        self,
        parser,
        repo: MessageRepoPort,
        concession_service: Optional[ConcessionService] = None,
        llm: Optional[LLMPort] = None,
        history_limit=5,
    ):
        self.parser = parser
        self.repo = repo
        self.history_limit = history_limit
        self.llm = llm
        self.concession_service = concession_service or ConcessionService(
            llm=llm,
        )

    async def handle(self, message: str, conversation_id: Optional[int] = None):
        if conversation_id is None:
            topic, side = self.parser(message)
            return await self.start_conversation(topic, side, message)

        assert_no_topic_or_side_markers(message)
        return await self.continue_conversation(message, conversation_id)

    async def start_conversation(self, topic: str, side: str, message: str = None):
        conversation = await self.repo.create_conversation(topic=topic, side=side)

        await self.repo.add_message(conversation_id=conversation.id, role="user", text=message)

        reply = await self.llm.generate(conversation=conversation)

        await self.repo.add_message(conversation_id=conversation.id, role="bot", text=reply)

        return {
            "conversation_id": conversation.id,
            "message": await self.repo.last_messages(conversation_id=conversation.id, limit=self.history_limit * 2),
        }

    async def continue_conversation(self, message: str, conversation_id: int):
        conversation = await self.repo.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise ConversationNotFound("conversation_id not found or expired")

        if conversation.expires_at <= datetime.now(timezone.utc):
            raise ConversationExpired("conversation_id expired")

        cid = conversation.id
        await self.repo.touch_conversation(conversation_id=cid)
        await self.repo.add_message(conversation_id=cid, role="user", text=message)

        full_history = await self.repo.all_messages(conversation_id=cid)

        reply = await self.concession_service.analyze_conversation(
            messages=full_history,
            side=conversation.side,
            conversation_id=conversation_id,
        )

        await self.repo.add_message(conversation_id=cid, role="bot", text=reply)

        return {
            "conversation_id": cid,
            "message": await self.repo.last_messages(conversation_id=cid, limit=self.history_limit * 2),
        }
