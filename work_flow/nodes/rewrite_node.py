from llm.models import get_chat_model
from llm.prompts import build_rewrite_message
from work_flow.copy_rag_document_state import CopyRagDocumentState




async def rewrite_node(state:CopyRagDocumentState)->CopyRagDocumentState:
    question = state["question"]
    history = state.get("history", [])

    messages = build_rewrite_message(question,history)

    response = await get_chat_model().ainvoke(messages)

    update:CopyRagDocumentState = {"question_rewrite":response.content}

    return update