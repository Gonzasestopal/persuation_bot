
import asyncio

from fastapi import APIRouter, Depends, Response, status

from app.api.dto import ConversationOut, MessageOut
from app.api.requests import MessageIn
from app.domain.errors import LLMTimeout  # domain-level
from app.factories import get_service
from app.settings import settings

router = APIRouter()


@router.post("/messages", response_model=ConversationOut)
async def post_messages(message: MessageIn, response: Response, service=Depends(get_service)):
    is_new = message.conversation_id is None
    try:
        result = await asyncio.wait_for(
            service.handle(conversation_id=message.conversation_id, message=message.message),
            timeout=settings.REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        # normalize transport-level timeout to domain error; global handler returns 503
        raise LLMTimeout("response generation timed out")

    response.status_code = status.HTTP_201_CREATED if is_new else status.HTTP_200_OK
    return ConversationOut(
        conversation_id=result["conversation_id"],
        message=[MessageOut(role=m.role, message=m.message) for m in result["message"]],
    )
