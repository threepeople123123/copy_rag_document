from typing import AsyncIterator
from uuid import UUID, uuid4

from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.chat import SseIterator
from core.logging import get_logger
from core.redis_client import redis_client
from llm.models import get_chat_model
from llm.prompts import build_oa_messages, build_compress_messages
from work_flow.copy_rag_document_state import Message, CopyRagDocumentState, MessageRole
from work_flow.graph import get_rag_graph

_message_adapter = TypeAdapter(list[Message])

logger = get_logger(__name__)



class ChatService:

    def __init__(self, session: AsyncSession):
        self.session = session


    async def stream_answer(self,conversation_id:UUID,question:str)->AsyncIterator[SseIterator]:

        try:
            history_json = redis_client.get(str(conversation_id))

            update:CopyRagDocumentState = {"question":question}
            update["conversation_id"] = conversation_id
            await load_content(update)
            megs : list[Message] = []
            if history_json:
                megs = TypeAdapter(list[Message]).validate_json(history_json)
                last_megs = megs[-20:]
                update["history"] = last_megs
            trace_id = UUID(str(uuid4())).hex
            update["trace_id"] = trace_id

            # 图里的节点全是 async 函数,必须用异步 API ainvoke/astream;
            # 用同步 invoke() 会报 "No synchronous function provided to ..."
            final_state = await get_rag_graph().ainvoke(update)

            answer_parts : list[str] = []
            async for chunk in stream_chat(final_state):
                answer_parts.append(chunk.data["data"])
                yield chunk
            answer = "".join(answer_parts)

            user_message = Message(role=MessageRole.USER, content=question)
            assistant_message = Message(role=MessageRole.ASSISTANT,content=answer)
            megs.append(user_message)
            megs.append(assistant_message)
            redis_client.set(str(f"chat_history:{conversation_id}"),_message_adapter.dump_json(megs).decode("utf-8"))
        except Exception as e:
            logger.error("ai生成失败:%s",e)
            yield SseIterator(
                data={"data":str(e)},
                event="error",
            )



async def stream_chat(copy_rag_document_state:CopyRagDocumentState)->AsyncIterator[SseIterator]:
    history = copy_rag_document_state.get("history", [])
    intent = copy_rag_document_state["intent"]
    chunks = copy_rag_document_state.get("retrieve",[])
    history_compress = copy_rag_document_state.get("history_compress","暂无")
    question_rewrite = copy_rag_document_state.get("question_rewrite", copy_rag_document_state["question"])
    messages = build_oa_messages(intent,chunks,question_rewrite,history,history_compress)
    async for stream in get_chat_model().astream(messages):
        text = stream.content
        if text:
            ses_data = SseIterator(
                data={"data":text},
                event="data"
            )
            yield ses_data


async def load_content(state:CopyRagDocumentState):
    conversation_id = state["conversation_id"]
    history_json = redis_client.get(str(f"chat_history:{conversation_id}"))

    compress_history = redis_client.get(str(f"chat_compress:{conversation_id}"))
    if history_json:
        megs = TypeAdapter(list[Message]).validate_json(history_json)
        if len(megs) >= 30:
            # 后面20条数据放入历史消息中
            state["history"] = megs[-20:]
            # 前面数据进行压缩
            front_megs = megs
            compress_history_json = _message_adapter.dump_json(front_megs).decode("utf-8")
            messages = build_compress_messages(str(compress_history),compress_history_json)
            response = await get_chat_model().ainvoke(messages)
            state["history_compress"] = response.content
            redis_client.set(str(f"chat_compress:{conversation_id}"), response.content)
        else:
            last_megs = megs[-30:]
            state["history"] = last_megs




