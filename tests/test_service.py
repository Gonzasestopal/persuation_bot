from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.message_service import MessageService


@pytest.fixture
def repo():
    return SimpleNamespace(
        create_conversation=AsyncMock(return_value=42),
        get_conversation=AsyncMock(return_value={"id": 123, "topic": "X", "side": "con"}),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(return_value=[
            {"role": "user", "message": "I firmly believe..."},
            {"role": "bot",  "message": "bot reply"},
        ]),
    )

@pytest.mark.asyncio
async def test_new_conversation(repo):
    parser = Mock(return_value=("X", "con"))
    svc = MessageService(parser=parser, repo=repo)
    svc.start_conversation = AsyncMock(return_value={"ok": "start"})
    svc.continue_conversation = AsyncMock()

    out = await svc.handle(message="Topic: X, Side: con")

    parser.assert_called_once_with("Topic: X, Side: con")
    # Your method signature is positional: (topic, side)
    svc.start_conversation.assert_awaited_once_with("X", "con")
    svc.continue_conversation.assert_not_called()
    assert out == {"ok": "start"}


@pytest.mark.asyncio
async def test_continue_conversation(repo):
    parser = Mock(side_effect=AssertionError("parser must not be called on continue"))

    svc = MessageService(parser=parser, repo=repo)
    svc.start_conversation = AsyncMock()
    svc.continue_conversation = AsyncMock(return_value={"ok": "continue"})

    out = await svc.handle(message="I firmly believe...", conversation_id=123)

    parser.assert_not_called()
    svc.continue_conversation.assert_awaited_once_with("I firmly believe...", 123)
    svc.start_conversation.assert_not_called()
    assert out == {"ok": "continue"}


@pytest.mark.asyncio
async def test_new_conversation_invalid_message(repo):
    parser = Mock()
    parser.side_effect = ValueError("message must contain Topic: and Side: fields")
    service = MessageService(parser=parser, repo=repo)
    service.start_conversation = AsyncMock()
    with pytest.raises(ValueError, match="message must contain Topic: and Side: fields"):
        await service.handle(message="Message missing params")

    service.start_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_continue_conversation_new_topic_or_side(repo):
    parser = Mock()
    service = MessageService(parser=parser, repo=repo)
    service.continue_conversation = AsyncMock()
    with pytest.raises(ValueError, match="topic/side must not be provided when continuing a conversation"):
        await service.handle(message="Topic: X, Side: PRO", conversation_id=123)

    service.continue_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_continue_rejects_topic_marker(repo):
    parser = Mock()
    service = MessageService(parser=parser, repo=repo)
    with pytest.raises(ValueError, match="must not be provided"):
        await service.handle(message="Topic: Cats. anyway...", conversation_id=1)


@pytest.mark.asyncio
async def test_continue_rejects_side_marker(repo):
    parser = Mock()
    service = MessageService(parser=parser, repo=repo)
    with pytest.raises(ValueError, match="must not be provided"):
        await service.handle(message="Side: PRO. I think...", conversation_id=1)


@pytest.mark.asyncio
async def test_continue_allows_normal_text_and_no_parser(repo):
    parser = Mock(side_effect=AssertionError("parser must not be called"))
    service = MessageService(parser=parser, repo=repo)
    service.continue_conversation = AsyncMock()
    await service.handle(message="We worked alongside: our peers", conversation_id=7)
    service.continue_conversation.assert_called()


@pytest.mark.asyncio
async def test_continue_with_empty_message(repo):
    parser = Mock(side_effect=AssertionError("parser must not be called"))
    service = MessageService(parser=parser, repo=repo)
    with pytest.raises(ValueError, match="must not be empty"):
        await service.handle(message="", conversation_id=7)
