# tests/test_conversation_route_dto.py
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dto import ConversationOut, MessageOut
from app.domain.models import Message


def build_app():
    app = FastAPI()

    @app.post('/conversations', response_model=ConversationOut)
    def create_conv():
        domain_msgs = [
            Message(role='user', message='u1', created_at=datetime.now(timezone.utc)),
            Message(role='bot', message='b1', created_at=datetime.now(timezone.utc)),
        ]
        return ConversationOut(
            conversation_id=123,
            message=[MessageOut(role=m.role, message=m.message) for m in domain_msgs],
        )

    return app


def test_route_hides_created_at_in_json_response():
    app = build_app()
    client = TestClient(app)

    resp = client.post('/conversations')
    assert resp.status_code == 200

    data = resp.json()
    assert data['conversation_id'] == 123
    assert 'message' in data and len(data['message']) == 2

    # DTOs should not include created_at
    assert data['message'][0] == {'role': 'user', 'message': 'u1'}
    assert data['message'][1] == {'role': 'bot', 'message': 'b1'}
    assert 'created_at' not in data['message'][0]
    assert 'created_at' not in data['message'][1]
