from pydantic import BaseModel


class MessageOut(BaseModel):
    role: str
    message: str


class ConversationOut(BaseModel):
    conversation_id: int
    message: list[MessageOut]
