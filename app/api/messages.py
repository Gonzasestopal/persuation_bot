
import asyncio

from fastapi import APIRouter, Depends
from app.api.requests import MessageIn
from app.services.message_service import MessageService
from app.settings import settings
from starlette.responses import JSONResponse


router = APIRouter()

def get_service() -> MessageService:
    return MessageService()

@router.post("/messages", status_code=201)
async def start_conversation(message: MessageIn, service=Depends(get_service)):
    try:
        return await asyncio.wait_for(
            service.handle(
                conversation_id=message.conversation_id,
                message=message.message,
            ),
            timeout=settings.REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=503, content={"detail": "response generation timed out"})
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    except KeyError:
        return JSONResponse(status_code=404, content={"detail": "conversation_id not found or expired"})
