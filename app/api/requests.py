from pydantic import BaseModel
from typing import Optional


class MessageIn(BaseModel):
    message: str
    conversation_id: Optional[int] = None
