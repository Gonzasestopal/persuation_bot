from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_start_conversation():
    payload = {
        "conversation_id": None,
        "message": "Topic: X. Side: PRO.",
    }
    r = client.post("/messages", json=payload)
    assert r.status_code == 201
    response = r.json()
    assert isinstance(response["conversation_id"], int)
    assert isinstance(response["message"], list)

def test_empty_message_422():
    payload = {"conversation_id": None, "message": ""}
    r = client.post("/messages", json=payload)
    assert r.status_code == 422
    assert "must not be empty" in r.json()["detail"]

def test_start_with_conversation_id_422():
    payload = {"conversation_id": 123, "message": "hello"}
    r = client.post("/messages", json=payload)
    assert r.status_code == 422
    assert "conversation_id must be null" in r.json()["detail"]

def test_not_found_conversation():
    payload = {"conversation_id": 9999, "message": "hi"}
    r = client.post("/messages", json=payload)
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()

def test_timeout():
    test_client = TestClient(app)
    r = test_client.post("/messages", json={"conversation_id": None, "message": "x"})
    assert r.status_code == 503
    assert "timed out" in r.json()["detail"].lower()
