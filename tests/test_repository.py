import datetime as dt

import pytest

from tests.fakes import InMemoryRepo


@pytest.fixture
def repo():
    return InMemoryRepo()


@pytest.mark.asyncio
async def test_create_and_get_conversation(repo):
    repo = InMemoryRepo()
    cid = await repo.create_conversation(topic='T', side='pro')
    assert isinstance(cid, int)

    row = await repo.get_conversation(cid)
    assert row and row["conversation_id"] == cid
    assert row["topic"] == "T"
    assert row["side"] == "pro"
    assert isinstance(row["expires_at"], dt.datetime)


@pytest.mark.asyncio
async def test_touch_conversation_bumps_expiry(repo):
    cid = await repo.create_conversation(topic="X", side="con")
    before = (await repo.get_conversation(cid))["expires_at"]
    await repo.touch_conversation(cid)
    after = (await repo.get_conversation(cid))["expires_at"]
    assert after > before


@pytest.mark.asyncio
async def test_last_messages_are_per_conversation(repo):
    a = await repo.create_conversation(topic="A", side="pro")
    b = await repo.create_conversation(topic="B", side="con")
    await repo.add_message(a, role="user", text="a1")
    await repo.add_message(b, role="user", text="b1")

    out_a = await repo.last_messages(a, limit=10)
    out_b = await repo.last_messages(b, limit=10)
    assert [m["message"] for m in out_a] == ["a1"]
    assert [m["message"] for m in out_b] == ["b1"]


@pytest.mark.asyncio
async def test_add_and_list_last_messages_order_and_limit(repo):
    cid = await repo.create_conversation(topic="T", side="pro")
    # add 6 messages
    for i in range(6):
        await repo.add_message(cid, role="user" if i % 2 == 0 else "bot", text=f"m{i}")

    out = await repo.last_messages(cid, limit=5)
    assert len(out) == 5
    # ascending by created_at (oldest→newest)
    times = [m["created_at"] for m in out]
    assert times == sorted(times)
    # only fields we expect
    assert {"role", "message", "created_at"} <= out[0].keys()
