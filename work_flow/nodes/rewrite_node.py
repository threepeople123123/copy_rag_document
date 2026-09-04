from uuid import uuid4

from core.langfuse import langfuse_client, send_message_to_langfuse
from llm.models import get_chat_model
from llm.prompts import build_rewrite_message
from work_flow.copy_rag_document_state import CopyRagDocumentState




async def rewrite_node(state:CopyRagDocumentState)->CopyRagDocumentState:
    question = state["question"]
    history = state.get("history", [])

    messages = build_rewrite_message(question,history)

    response = await get_chat_model().ainvoke(messages)

    update:CopyRagDocumentState = {"question_rewrite":response.content}

    trace_id = state.get("trace_id",str(uuid4().hex))

    update["trace_id"] = trace_id

    # 发送langfuse
    await send_message_to_langfuse(trace_id,"rewrite_node",{"question":question},{"question_rewrite":response.content})

    return update