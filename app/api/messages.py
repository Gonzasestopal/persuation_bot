
from fastapi import APIRouter
from app.api.requests import MessageIn


router = APIRouter()

@router.post("/messages", status_code=201)
async def start_conversation(message: MessageIn):
    pass
