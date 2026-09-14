from typing import AsyncIterable

from fastapi.routing import APIRouter

from react_agent.react_agent_graph import get_react_agent_graph

route= APIRouter(prefix="/react_agent",tags=["react_agent"])

@route.post("/chat")
async def chat(question: str, conversation_id: str) -> AsyncIterable[dict]:
    config = {"configurable": {"thread_id": conversation_id}}
    init = {"question": question, "conversation_id": conversation_id}

    # 首次运行：跑到 human_node 的 interrupt 暂停；updates 里会带 __interrupt__
    async for result in get_react_agent_graph().astream(input=init, config=config, stream_mode="updates"):
        if result:
            yield result
