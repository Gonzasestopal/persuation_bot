
import asyncio

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from app.api.requests import MessageIn
from app.deps import get_repo
from app.domain.parser import parse_topic_side
from app.llm.dummy import DummyLLMAdapter
from app.llm.interface import LLMAdapterInterface
from app.services.message_service import MessageService
from app.settings import settings

router = APIRouter()


def get_llm() -> LLMAdapterInterface:
    return DummyLLMAdapter()


def get_service(
    repo=Depends(get_repo),
    llm=Depends(get_llm)
) -> MessageService:
    return MessageService(
        parser=parse_topic_side,
        repo=repo,
        llm=llm,
    )


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
