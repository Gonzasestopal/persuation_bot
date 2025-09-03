import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service
from app.domain.errors import ConversationNotFound, InvalidStartMessage
from app.main import app
from app.settings import settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


class FakeOK:
    async def handle(self, **_):
        return {'conversation_id': 1, 'message': []}


class FakeNotFound:
    async def handle(self, **_):
        raise ConversationNotFound('conversation not found')


class FakeMissingMessage:
    async def handle(self, **_):
        raise InvalidStartMessage('message must not be empty.')


class FakeBadInput:
    async def handle(self, **_):
        raise InvalidStartMessage('conversation_id must be null when starting')


class FakeSlow:
    async def handle(self, **_):
        await asyncio.sleep(1)
        return {'conversation_id': 1, 'message': []}


def test_start_conversation():
    app.dependency_overrides[get_service] = lambda: FakeOK()
    payload = {
        'conversation_id': None,
        'message': 'Topic: X. Side: PRO.',
    }
    r = client.post('/messages', json=payload)
    assert r.status_code == 201
    response = r.json()
    assert isinstance(response['conversation_id'], int)
    assert isinstance(response['message'], list)


def test_empty_message_422():
    app.dependency_overrides[get_service] = lambda: FakeMissingMessage()
    payload = {'conversation_id': None, 'message': ''}
    r = client.post('/messages', json=payload)
    assert r.status_code == 422
    assert 'must not be empty' in r.json()['detail']


def test_start_with_conversation_id_422():
    app.dependency_overrides[get_service] = lambda: FakeBadInput()
    payload = {'conversation_id': 123, 'message': 'hello'}
    r = client.post('/messages', json=payload)
    assert r.status_code == 422
    assert 'conversation_id must be null' in r.json()['detail']


def test_not_found_conversation():
    app.dependency_overrides[get_service] = lambda: FakeNotFound()
    payload = {'conversation_id': 9999, 'message': 'hi'}
    r = client.post('/messages', json=payload)
    assert r.status_code == 404
    assert 'not found' in r.json()['detail'].lower()


def test_timeout(monkeypatch):
    app.dependency_overrides[get_service] = lambda: FakeSlow()
    monkeypatch.setattr(settings, 'REQUEST_TIMEOUT_S', 0.05, raising=False)
    r = client.post('/messages', json={'conversation_id': None, 'message': 'x'})
    assert r.status_code == 503
    assert 'timed out' in r.json()['detail'].lower()
