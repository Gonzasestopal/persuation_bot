from pydantic import BaseModel
from typing import Optional


class MessageIn(BaseModel):
    message: Optional[str] = None
    conversation_id: Optional[int] = None
