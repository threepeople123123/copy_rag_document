from typing import AsyncIterable

from fastapi import FastAPI, APIRouter
from fastapi.sse import EventSourceResponse

from api.schemas.chat import SseIterator, ChatRequest
from db.session import DbSession
from service.chat_service import ChatService

route = APIRouter(prefix="/chat",tags=["chat"])



@route.post("/stream_answer",operation_id="streamChat",response_class=EventSourceResponse)
async def stream_answer(payload:ChatRequest,session: DbSession)->AsyncIterable[SseIterator]:
    chat_service = ChatService(session)

    async for chunk in chat_service.stream_answer(question=payload.question,conversation_id=payload.conversation_id):
        yield chunk

