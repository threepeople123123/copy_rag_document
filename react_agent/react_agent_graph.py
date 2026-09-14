from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from core.config import settings
from core.logging import get_logger

from react_agent.nodes.agent_node import agent_node, route_after_agent
from react_agent.nodes.finish_node import finish_node
from react_agent.nodes.tool_call_node import tool_call_node
from react_agent.react_state import ReActState

logger = get_logger(__name__)


def _build_checkpointer():
    sync_url = settings.date_source_url
    if sync_url.startswith("postgresql+asyncpg://"):
        sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    pool = None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            sync_url,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        saver = PostgresSaver(pool)
        saver.setup()
        return saver
    except Exception as exc:
        logger.warning("Postgres checkpointer 初始化失败，降级 memory：%s", exc)
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
    return MemorySaver()


def _build_graph():
    builder = StateGraph(ReActState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_call_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finish": "finish"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=_build_checkpointer())


_react_agent_graph = _build_graph()


def get_react_agent_graph():
    """对外暴露编译好的 ReAct agent 图。"""
    return _react_agent_graph


async def run_react_agent(question: str,
                          history: list | None = None,
                          conversation_id: str | None = None) -> dict:
    """统一入口：跑一次 ReAct agent，返回最终 state（含 final_answer）。"""
    init: dict = {"question": question}
    if history:
        init["history"] = history
    if conversation_id:
        init["conversation_id"] = conversation_id

    config = {"configurable": {"thread_id": conversation_id or "default"}}
    return await _react_agent_graph.ainvoke(init, config=config)
