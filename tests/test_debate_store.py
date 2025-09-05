# tests/test_debate_store_port.py
from copy import deepcopy

import pytest

from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
from app.domain.concession_policy import DebateState


@pytest.fixture
def store():
    return InMemoryDebateStore()


def test_get_returns_none_when_absent(store):
    assert store.get(9999) is None


def test_create_then_get_returns_copy(store):
    conv_id = 1
    created = store.create(conv_id, stance='pro', lang='es', topic='ok')
    assert isinstance(created, DebateState)
    assert created.stance == 'pro'
    assert created.lang == 'es'

    fetched = store.get(conv_id)
    assert fetched is not created  # copy, not same object
    assert fetched == created  # same values


def test_create_existing_raises_value_error(store):
    conv_id = 2
    store.create(conv_id, stance='con', lang='es', topic='god exists')
    with pytest.raises(ValueError):
        store.create(conv_id, stance='con', lang='es', topic='god exists')


def test_get_returns_deep_copy_not_live_reference(store):
    conv_id = 3
    s1 = store.create(conv_id, stance='pro', lang='es', topic='god exists')

    # mutate returned object without saving
    s1.positive_judgements += 1
    s1.match_concluded = True

    # store must still hold original values
    s_again = store.get(conv_id)
    assert s_again.positive_judgements == 0
    assert s_again.match_concluded is False


def test_save_persists_changes(store):
    conv_id = 4
    s = store.create(conv_id, stance='pro', lang='es', topic='god exists')
    s.positive_judgements = 2
    s.assistant_turns = 5
    store.save(conv_id, s)

    from_store = store.get(conv_id)
    assert from_store.positive_judgements == 2
    assert from_store.assistant_turns == 5


def test_update_applies_fn_and_returns_copy(store):
    conv_id = 5
    store.create(conv_id, stance='pro', lang='es', topic='god exists')

    def bump_and_conclude(st: DebateState):
        st.positive_judgements += 1
        st.match_concluded = True

    updated = store.update(conv_id, bump_and_conclude)
    again = store.get(conv_id)

    assert updated is not again
    assert updated.positive_judgements == 1
    assert updated.match_concluded is True
    assert again.positive_judgements == 1
    assert again.match_concluded is True


def test_update_missing_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.update(404, lambda s: None)


def test_save_overwrites_existing_state(store):
    conv_id = 6
    base = store.create(conv_id, stance='pro', lang='es', topic='god exists')

    local = deepcopy(base)
    local.positive_judgements = 7
    local.assistant_turns = 3
    local.match_concluded = True

    store.save(conv_id, local)

    after = store.get(conv_id)
    assert after.positive_judgements == 7
    assert after.assistant_turns == 3
    assert after.match_concluded is True


def test_copies_are_independent(store):
    conv_id = 7
    store.create(conv_id, stance='con', lang='es', topic='god exists')

    a = store.get(conv_id)
    b = store.get(conv_id)
    assert a is not b

    a.positive_judgements = 10
    # not saved; store and b remain unchanged
    b_again = store.get(conv_id)
    assert b_again.positive_judgements == 0
    assert b.positive_judgements == 0
