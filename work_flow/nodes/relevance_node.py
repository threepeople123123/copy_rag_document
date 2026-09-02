from dataclasses import dataclass

from pydantic import Field, BaseModel

from llm.models import get_chat_model
from llm.prompts import build_relevance_messages
from work_flow.copy_rag_document_state import CopyRagDocumentState


@dataclass
class Relevance(BaseModel):
    relevance_score: float= Field(..., description="打分字段赋值，如果低于")
    relevance_reason:str = Field(..., description="如此打分的理由")

async def relevance_node(state:CopyRagDocumentState)->CopyRagDocumentState:
    intent = state["intent"]
    retrieve = state.get("retrieve",[])
    question_rewrite = state["question_rewrite"]
    history = state.get("history", [])

    update:CopyRagDocumentState = {}

    # 规则查询或者流程查询，才会检查相关性，如果不是直接生成答案
    if intent =="rules_regulations" or intent == "work_flow":
        messages = build_relevance_messages(question_rewrite,history,retrieve)
        model_with_structure = get_chat_model().with_structured_output(Relevance)
        relevance_cls =  await model_with_structure.ainvoke(messages)
        relevance_score = relevance_cls.relevance_score
        relevance_reason = relevance_cls.relevance_reason
        update["relevance_score"] = relevance_score
        update["relevance_reason"] = relevance_reason
    return update