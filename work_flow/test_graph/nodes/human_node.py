from langgraph.types import interrupt

from work_flow.test_graph.test_state import TestState


async def human_node(state:TestState):
    interrupt("快速回复")