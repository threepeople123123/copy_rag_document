from langgraph.constants import START, END
from langgraph.graph import StateGraph
from work_flow.copy_rag_document_state import CopyRagDocumentState
from work_flow.nodes.plan_node import plan_node
from work_flow.nodes.relevance_node import relevance_node
from work_flow.nodes.intent_node import intent_node
from work_flow.nodes.question_node import question_node
from work_flow.nodes.retrieve_node import retrieve_node
from work_flow.nodes.rewrite_node import rewrite_node


def _after_intent_node(state:CopyRagDocumentState)->str:
    intent = state["intent"]
    if intent == "small_talk":
        return "end"
    elif intent == "rules_regulations" or intent == "work_flow":
        return "question_node"
    return "end"


def _after_relevance_node(state: CopyRagDocumentState) -> str:
    cycle_count = state["cycle_count"]
    if cycle_count > 3:
        return "end"
    return "retrieve_node"




def _build_graph():
    builder = StateGraph(CopyRagDocumentState)

    builder.add_node("rewrite_node", rewrite_node)
    builder.add_node("intent_node", intent_node)
    builder.add_node("question_node", question_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("relevance_node", relevance_node)
    builder.add_node("plan_node", plan_node)

    builder.add_edge(START, "rewrite_node")
    builder.add_edge("rewrite_node","intent_node")
    builder.add_conditional_edges("intent_node",
                                  _after_intent_node,
                                  {"end":END,"question_node":"question_node"}
    )
    builder.add_edge("question_node","retrieve_node")
    builder.add_edge("retrieve_node","relevance_node")
    builder.add_conditional_edges("relevance_node",
                                  _after_relevance_node,
                                  {"end":END,"retrieve_node":"retrieve_node"}
    )

    return builder.compile()


_rag_graph = _build_graph()


def get_rag_graph():
    """对外暴露已编译好的子图；模块加载时一次编译，请求里直接复用。"""
    return _rag_graph