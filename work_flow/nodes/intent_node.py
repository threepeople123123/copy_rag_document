from pydantic import BaseModel, Field

from pydantic import BaseModel, Field

from core.langfuse import send_message_to_langfuse
from llm.models import get_structured_agent
from llm.prompts import build_route_message
from work_flow.copy_rag_document_state import CopyRagDocumentState


class RouteEnum(BaseModel):
    """一部带有详细信息的电影。"""
    route:str=Field(..., description="根据用户问题，进行判断，智能返回三种数据：small_talk,rules_regulations,work_flow")


async def intent_node(state:CopyRagDocumentState)->CopyRagDocumentState:
    question_rewrite = state["question_rewrite"]

    history = state.get("history", [])

    trace_id = state["trace_id"]

    messages = build_route_message(question_rewrite,history)

    agent_with_structure = get_structured_agent(RouteEnum)
    result = await agent_with_structure.ainvoke({
        "messages": messages,
    })
    question_diverse = result["structured_response"]

    replace_after = question_diverse.route.replace('"', '')
    intent = replace_after.replace("'", '')

    # 发送langfuse消息
    await send_message_to_langfuse(trace_id,"intent_node",{"question_rewrite":question_rewrite},{"intent":intent},"evaluator")

    return {"intent":intent}