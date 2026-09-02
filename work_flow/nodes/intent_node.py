from enum import Enum

from pydantic import BaseModel, Field

from llm.models import get_chat_model
from llm.prompts import build_route_message
from work_flow.copy_rag_document_state import CopyRagDocumentState


class RouteEnum(BaseModel):
    """一部带有详细信息的电影。"""
    route:str=Field(..., description="根据用户问题，进行判断，智能返回三种数据：small_talk,rules_regulations,work_flow")


async def intent_node(state:CopyRagDocumentState)->CopyRagDocumentState:
    question_rewrite = state["question_rewrite"]

    history = state.get("history", [])

    messages = build_route_message(question_rewrite,history)

    model_with_structure = get_chat_model().with_structured_output(RouteEnum)
    response = await model_with_structure.ainvoke(messages)

    replace_after = response.route.replace('"', '')
    route = replace_after.replace("'", '')

    return {"intent":route}