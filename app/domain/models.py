from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.enums import Stance


class Conversation(BaseModel):
    id: int
    topic: str
    stance: Stance
    expires_at: datetime


class Message(BaseModel):
    role: str = Field(pattern='^(user|bot)$')
    message: str
    created_at: datetime = None
