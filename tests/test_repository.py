import datetime as dt

import pytest

from app.domain.models import Conversation, Message
from tests.fakes import InMemoryRepo


@pytest.fixture
def repo():
    return InMemoryRepo()


@pytest.mark.asyncio
async def test_create_and_get_conversation(repo):
    repo = InMemoryRepo()
    conv = await repo.create_conversation(topic='T', side='pro')
    assert isinstance(conv, Conversation)

    retrieved_conv = await repo.get_conversation(conv.id)
    assert retrieved_conv and retrieved_conv.id == conv.id
    assert retrieved_conv.topic == "T"
    assert retrieved_conv.side == "pro"
    assert isinstance(retrieved_conv.expires_at, dt.datetime)


@pytest.mark.asyncio
async def test_touch_conversation_bumps_expiry(repo):
    conv = await repo.create_conversation(topic="X", side="con")
    before = (await repo.get_conversation(conv.id)).expires_at
    await repo.touch_conversation(conv.id)
    after = (await repo.get_conversation(conv.id)).expires_at
    assert after > before


@pytest.mark.asyncio
async def test_last_messages_are_per_conversation(repo):
    a = await repo.create_conversation(topic="A", side="pro")
    b = await repo.create_conversation(topic="B", side="con")
    await repo.add_message(a.id, role="user", text="a1")
    await repo.add_message(b.id, role="user", text="b1")

    out_a = await repo.last_messages(a.id, limit=10)
    out_b = await repo.last_messages(b.id, limit=10)
    assert [m.message for m in out_a] == ["a1"]
    assert [m.message for m in out_b] == ["b1"]


@pytest.mark.asyncio
async def test_add_and_list_last_messages_order_and_limit(repo):
    conv = await repo.create_conversation(topic="T", side="pro")
    # add 6 messages
    for i in range(6):
        await repo.add_message(conv.id, role="user" if i % 2 == 0 else "bot", text=f"m{i}")

    out = await repo.last_messages(conv.id, limit=5)
    assert len(out) == 5
    # ascending by created_at (oldest→newest)
    times = [m.created_at for m in out]
    assert times == sorted(times)
    # only fields we expect
    assert {"role", "message", "created_at"} <= out[0].model_dump().keys()


@pytest.mark.asyncio
async def test_last_messages_window_slides_on_new_message(repo):
    conv = await repo.create_conversation(topic="T", side="pro")

    for i in range(5):
        await repo.add_message(conv.id, role="user", text=f"m{i}")

    out1 = await repo.last_messages(conv.id, limit=5)
    assert [m.message for m in out1] == ["m0", "m1", "m2", "m3", "m4"]

    await repo.add_message(conv.id, role="user", text="m5")
    out2 = await repo.last_messages(conv.id, limit=5)
    assert [m.message for m in out2] == ["m1", "m2", "m3", "m4", "m5"]

    t2 = [m.created_at for m in out2]
    assert t2 == sorted(t2)


@pytest.mark.asyncio
async def test_user_bot_order_preserved_with_same_timestamp(repo):
    conv = await repo.create_conversation(topic="T", side="pro")
    ts = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)

    # Same timestamp, but inserted user then bot repeatedly
    await repo.add_message(conv.id, role="user", text="u0", created_at=ts)
    await repo.add_message(conv.id, role="bot",  text="b0", created_at=ts)
    await repo.add_message(conv.id, role="user", text="u1", created_at=ts)
    await repo.add_message(conv.id, role="bot",  text="b1", created_at=ts)
    await repo.add_message(conv.id, role="user", text="u2", created_at=ts)
    await repo.add_message(conv.id, role="bot",  text="b2", created_at=ts)

    out = await repo.last_messages(conv.id, limit=6)
    assert [(m.role, m.message) for m in out] == [
        ("user", "u0"), ("bot", "b0"),
        ("user", "u1"), ("bot", "b1"),
        ("user", "u2"), ("bot", "b2"),
    ]
