from typing import Optional

from pydantic import BaseModel


class MessageIn(BaseModel):
    message: Optional[str] = None
    conversation_id: Optional[int] = None
