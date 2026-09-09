from llm.models import get_chat_model
from llm.test_agent_prompt import build_intention_messages
from work_flow.test_graph.test_state import TestState

async def intention_node(state:TestState)->TestState:

    # 问题
    question = state["question"]
    history = state.get("history",[])
    message = build_intention_messages(question,history)
    response  = await get_chat_model().ainvoke(message)
    content = response.content
    if content:
        content = content.replace("`","")
        content = content.replace("'", "")
        content = content.replace('"', "")
    return {"intention":content}