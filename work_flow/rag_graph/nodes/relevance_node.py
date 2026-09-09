from dataclasses import dataclass

from pydantic import Field, BaseModel

from core.langfuse import send_message_to_langfuse
from llm.models import get_structured_agent
from llm.prompts import build_relevance_messages
from work_flow.rag_graph.copy_rag_document_state import RagDocumentState


@dataclass
class Relevance(BaseModel):
    relevance_score: float= Field(..., description="打分字段赋值，如果低于")
    relevance_reason:str = Field(..., description="如此打分的理由")

async def relevance_node(state:RagDocumentState)->RagDocumentState:
    intent = state["intent"]
    retrieve = state.get("retrieve",[])
    question_rewrite = state["question_rewrite"]
    history = state.get("history", [])
    trace_id = state["trace_id"]

    update:RagDocumentState = {}

    # 规则查询或者流程查询，才会检查相关性，如果不是直接生成答案
    if intent =="rules_regulations" or intent == "work_flow":
        messages = build_relevance_messages(question_rewrite,history,retrieve)
        agent_with_structure = get_structured_agent(Relevance)
        result = await agent_with_structure.ainvoke({
            "messages": messages,
        })
        relevance_cls = result["structured_response"]
        relevance_score = relevance_cls.relevance_score
        relevance_reason = relevance_cls.relevance_reason
        update["relevance_score"] = relevance_score
        update["relevance_reason"] = relevance_reason
        input_message:dict = {"question_rewrite":question_rewrite,"history":str(history),"retrieve":str(retrieve)}
        output_message:dict = {"relevance_score":relevance_score,"relevance_reason":relevance_reason}

        # 发送langfuse消息
        await send_message_to_langfuse(trace_id,"relevance_node",input_message,output_message,"evaluator")

    return update