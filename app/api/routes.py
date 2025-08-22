
import asyncio

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from app.api.dto import ConversationOut, MessageOut
from app.api.requests import MessageIn
from app.factories import get_service
from app.settings import settings

router = APIRouter()


@router.post("/messages", status_code=201, response_model=ConversationOut)
async def start_conversation(message: MessageIn, service=Depends(get_service)):
    try:
        result = await asyncio.wait_for(
            service.handle(
                conversation_id=message.conversation_id,
                message=message.message,
            ),
            timeout=settings.REQUEST_TIMEOUT_S,
        )
        return ConversationOut(
            conversation_id=result["conversation_id"],
            message=[MessageOut(role=m.role, message=m.message) for m in result["message"]],
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=503, content={"detail": "response generation timed out"})
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    except KeyError:
        return JSONResponse(status_code=404, content={"detail": "conversation_id not found or expired"})
