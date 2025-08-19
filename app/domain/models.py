from datetime import datetime

from pydantic import BaseModel, Field


class Conversation(BaseModel):
    id: int
    topic: str
    side: str
    expires_at: datetime


class Message(BaseModel):
    role: str = Field(pattern="^(user|bot)$")
    message: str
    created_at: datetime = None
