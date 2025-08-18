from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from app.services.message_service import MessageService


@pytest.fixture
def repo():
    expired_time = datetime.now(timezone.utc) + timedelta(minutes=60)
    return SimpleNamespace(
        create_conversation=AsyncMock(return_value=42),  # not used here
        get_conversation=AsyncMock(return_value={"conversation_id": 123, "topic": "X", "side": "con", "expires_at": expired_time}),
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

    txt = "Topic: X, Side: con"
    out = await svc.handle(message=txt)

    parser.assert_called_once_with(txt)
    # Your method signature is positional: (topic, side)
    svc.start_conversation.assert_awaited_once_with("X", "con", txt)
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

    out = await svc.start_conversation(topic="X", side="con", message="Topic: X, Side: con")

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
        call(conversation_id=123, role="bot",  text="bot reply"),
    ])
    repo.last_messages.assert_awaited_once_with(conversation_id=123, limit=10)  # 5 pairs * 2
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

    repo.get_conversation.assert_awaited_once_with(conversation_id=9999)
    repo.touch_conversation.assert_not_called()
    repo.add_message.assert_not_called()
    repo.last_messages.assert_not_called()


@pytest.mark.asyncio
async def test_continue_conversation_respects_history_limit():
    expired_time = datetime.now(timezone.utc) + timedelta(minutes=60)
    repo = SimpleNamespace(
        create_conversation=AsyncMock(),
        get_conversation=AsyncMock(return_value={"conversation_id": 123, "topic": "X", "side": "con", "expires_at": expired_time}),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(return_value=[
            {"role": "user", "message": "hi"},
            {"role": "bot", "message": "bot reply"},
        ]),
    )

    parser = Mock(side_effect=AssertionError("parser must not be called"))
    svc = MessageService(parser=parser, repo=repo, history_limit=2)

    out = await svc.continue_conversation(message="hi", conversation_id=123)

    repo.get_conversation.assert_awaited_once_with(conversation_id=123)
    repo.touch_conversation.assert_awaited_once_with(conversation_id=123)
    repo.add_message.assert_has_awaits([
        call(conversation_id=123, role="user", text="hi"),
        call(conversation_id=123, role="bot",  text="bot reply"),
    ])
    # history_limit=2 → 2 * 2 = 4 messages window
    repo.last_messages.assert_awaited_once_with(conversation_id=123, limit=4)

    assert out == {
        "conversation_id": 123,
        "message": [
            {"role": "user", "message": "hi"},
            {"role": "bot", "message": "bot reply"},
        ],
    }

@pytest.mark.asyncio
async def test_continue_conversation_expired(repo):
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=1)
    repo.get_conversation.return_value = {"conversation_id": 123, "expires_at": expired_time}

    svc = MessageService(parser=Mock(), repo=repo)

    with pytest.raises(KeyError, match="expired"):
        await svc.continue_conversation("hello", 123)

    repo.touch_conversation.assert_not_called()
    repo.add_message.assert_not_called()


@pytest.mark.asyncio
async def test_start_conversation_calls_llm_and_stores_reply():
    repo = SimpleNamespace(
        create_conversation=AsyncMock(return_value=42),
        get_conversation=AsyncMock(),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(side_effect=[
            [{"role": "user", "message": "Topic: X, Side: con"}],
            [
                {"role": "user", "message": "Topic: X, Side: con"},
                {"role": "bot", "message": "Hello from LLM"},
            ],  # final return
        ]),
    )

    parser = Mock(return_value=("X", "con"))
    llm = AsyncMock()
    llm.generate.return_value = "Hello from LLM"

    svc = MessageService(parser=parser, repo=repo, llm=llm)

    out = await svc.start_conversation("X", "con", "Topic: X, Side: con")

    llm.generate.assert_awaited_once_with([
        {"role": "user", "message": "Topic: X, Side: con"},
    ])

    repo.add_message.assert_has_awaits([
        call(conversation_id=42, role="user", text="Topic: X, Side: con"),
        call(conversation_id=42, role="bot", text="Hello from LLM"),
    ])

    assert out["message"][-1]["message"] == "Hello from LLM"
