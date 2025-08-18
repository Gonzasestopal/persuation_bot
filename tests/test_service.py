from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from app.services.message_service import MessageService


@pytest.fixture
def repo():
    return SimpleNamespace(
        create_conversation=AsyncMock(return_value=42),  # not used here
        get_conversation=AsyncMock(return_value={"id": 123, "topic": "X", "side": "con"}),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(return_value=[
            {"role": "user", "message": "I firmly believe..."},
            {"role": "bot",  "message": "OK"},
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


@pytest.mark.asyncio
async def test_start_writes_messages_and_returns_window():
    repo = SimpleNamespace(
        create_conversation=AsyncMock(return_value=42),
        get_conversation=AsyncMock(),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(return_value=[
            {"role": "user", "message": "Topic: X, Side: con"},
            {"role": "bot",  "message": "bot reply"},
        ]),
    )

    parser = Mock(return_value=("X", "con"))
    svc = MessageService(parser=parser, repo=repo)

    out = await svc.start_conversation(topic="X", side="con")

    repo.create_conversation.assert_awaited_once_with(topic="X", side="con")
    repo.add_message.assert_has_awaits([
        call(conversation_id=42, role="user", text="Topic: X, Side: con"),
        call(conversation_id=42, role="bot",  text="bot reply"),
    ])
    repo.last_messages.assert_awaited_once_with(conversation_id=42, limit=10)
    assert out == {
        "conversation_id": 42,
        "message": [
            {"role": "user", "message": "Topic: X, Side: con"},
            {"role": "bot",  "message": "bot reply"},
        ],
    }


@pytest.mark.asyncio
async def test_continue_conversation_writes_and_returns_window(repo):
    parser = Mock(side_effect=AssertionError("parser must not be called on continue"))
    svc = MessageService(parser=parser, repo=repo, history_limit=5)

    out = await svc.continue_conversation(message="I firmly believe...", conversation_id=123)

    repo.get_conversation.assert_awaited_once_with(conversation_id=123)
    repo.touch_conversation.assert_awaited_once_with(conversation_id=123)
    repo.add_message.assert_has_awaits([
        call(conversation_id=123, role="user", text="I firmly believe..."),
        call(conversation_id=123, role="bot",  text="OK"),
    ])
    repo.last_messages.assert_awaited_once_with(123, limit=10)  # 5 pairs * 2
    assert out == {
        "conversation_id": 123,
        "message": [
            {"role": "user", "message": "I firmly believe..."},
            {"role": "bot",  "message": "OK"},
        ],
    }


@pytest.mark.asyncio
async def test_continue_conversation_unknown_id_raises_keyerror():
    repo = SimpleNamespace(
        create_conversation=AsyncMock(),
        get_conversation=AsyncMock(return_value=None),  # not found / expired
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(),
    )
    parser = Mock()
    svc = MessageService(parser=parser, repo=repo)

    with pytest.raises(KeyError):
        await svc.continue_conversation(message="hi", conversation_id=9999)

    repo.get_conversation.assert_awaited_once_with(9999)
    repo.touch_conversation.assert_not_called()
    repo.add_message.assert_not_called()
    repo.last_messages.assert_not_called()


@pytest.mark.asyncio
async def test_continue_conversation_respects_history_limit():
    repo = SimpleNamespace(
        create_conversation=AsyncMock(),
        get_conversation=AsyncMock(return_value={"id": 123, "topic": "X", "side": "con"}),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(return_value=[
            {"role": "user", "message": "hi"},
            {"role": "bot", "message": "hello"},
        ]),
    )

    parser = Mock(side_effect=AssertionError("parser must not be called"))
    svc = MessageService(parser=parser, repo=repo, history_limit=2)

    out = await svc.continue_conversation(message="hi", conversation_id=123)

    repo.get_conversation.assert_awaited_once_with(conversation_id=123)
    repo.touch_conversation.assert_awaited_once_with(conversation_id=123)
    repo.add_message.assert_has_awaits([
        call(conversation_id=123, role="user", text="hi"),
        call(conversation_id=123, role="bot",  text="hello"),
    ])
    # history_limit=2 → 2 * 2 = 4 messages window
    repo.last_messages.assert_awaited_once_with(123, limit=4)

    assert out == {
        "conversation_id": 123,
        "message": [
            {"role": "user", "message": "hi"},
            {"role": "bot", "message": "hello"},
        ],
    }
