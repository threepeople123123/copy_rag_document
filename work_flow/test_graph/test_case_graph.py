from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from core.config import settings
from core.logging import get_logger
from work_flow.test_graph.nodes.generate_test_node import generate_test_node
from work_flow.test_graph.nodes.human_node import human_node
from work_flow.test_graph.nodes.intention_node import intention_node
from work_flow.test_graph.nodes.requirement_parser_node import requirement_parser_node
from work_flow.test_graph.test_state import TestState

logger = get_logger(__name__)


def _build_checkpointer():
    # langgraph 的 PostgresSaver 底层用同步 psycopg，只认 postgresql:// 协议；
    # 项目里的 DATE_SOURCE_URL 是 SQLAlchemy 异步协议 postgresql+asyncpg://，需转成同步协议。
    sync_url = settings.date_source_url
    if sync_url.startswith("postgresql+asyncpg://"):
        sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    pool = None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        # 注意：from_conn_string 是 @contextmanager，不能直接 return 它；
        # 这里用连接池构造一个长期可复用的 saver（与编译后的图生命周期一致）。
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
    builder = StateGraph(TestState)

    builder.add_node("intention_node", intention_node)
    builder.add_node("requirement_parser_node", requirement_parser_node)
    builder.add_node("generate_test_node", generate_test_node)
    builder.add_node("human_node", human_node)

    builder.add_edge(START, "intention_node")
    builder.add_edge("intention_node", "requirement_parser_node")
    builder.add_edge("requirement_parser_node", "generate_test_node")
    builder.add_edge("generate_test_node", "human_node")
    builder.add_edge("human_node", END)

    return builder.compile(checkpointer=_build_checkpointer())


test_graph = _build_graph()


def get_test_graph():
    """对外暴露已编译好的子图；模块加载时一次编译，请求里直接复用。"""
    return test_graph
