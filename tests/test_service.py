from unittest.mock import AsyncMock, Mock

import pytest

from app.services.message_service import MessageService


@pytest.mark.asyncio
async def test_new_conversation():
    parser = Mock()
    parser.return_value = ('X', 'con')
    service = MessageService(parser=parser)
    service.start_conversation = AsyncMock()
    await service.handle(message="Topic: X, Side: con")
    service.start_conversation.assert_called()


@pytest.mark.asyncio
async def test_continue_conversation():
    parser = Mock()
    service = MessageService(parser=parser)
    service.continue_conversation = AsyncMock()
    await service.handle(conversation_id=123, message="I firmly believe that...")
    service.continue_conversation.assert_called()


@pytest.mark.asyncio
async def test_new_conversation_invalid_message():
    parser = Mock()
    parser.side_effect = ValueError("message must contain Topic: and Side: fields")
    service = MessageService(parser=parser)
    service.start_conversation = AsyncMock()
    with pytest.raises(ValueError, match="message must contain Topic: and Side: fields"):
        await service.handle(message="Message missing params")

    service.start_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_continue_conversation_new_topic_or_side():
    parser = Mock()
    service = MessageService(parser=parser)
    service.continue_conversation = AsyncMock()
    with pytest.raises(ValueError, match="topic/side must not be provided when continuing a conversation"):
        await service.handle(message="Topic: X, Side: PRO", conversation_id=123)

    service.continue_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_continue_rejects_topic_marker():
    service = MessageService(parser=Mock())
    with pytest.raises(ValueError, match="must not be provided"):
        await service.handle(message="Topic: Cats. anyway…", conversation_id=1)


@pytest.mark.asyncio
async def test_continue_rejects_side_marker():
    service = MessageService(parser=Mock())
    with pytest.raises(ValueError, match="must not be provided"):
        await service.handle(message="Side: PRO. I think…", conversation_id=1)


@pytest.mark.asyncio
async def test_continue_allows_normal_text_and_no_parser():
    parser = Mock(side_effect=AssertionError("parser must not be called"))
    service = MessageService(parser)
    service.continue_conversation = AsyncMock()
    await service.handle(message="We worked alongside: our peers", conversation_id=7)
    service.continue_conversation.assert_called()


@pytest.mark.asyncio
async def test_continue_with_empty_message():
    parser = Mock(side_effect=AssertionError("parser must not be called"))
    service = MessageService(parser)
    with pytest.raises(ValueError, match="must not be empty"):
        await service.handle(message="", conversation_id=7)
