from llm.models import get_agent
from llm.prompts import build_rewrite_message
from work_flow.copy_rag_document_state import CopyRagDocumentState




async def rewrite_node(state:CopyRagDocumentState)->CopyRagDocumentState:
    question = state["question"]
    history = state["history"]

    messages = build_rewrite_message(question,history)

    response = await get_agent().ainvoke(messages)

    update:CopyRagDocumentState = {"question_rewrite":response}

    return update