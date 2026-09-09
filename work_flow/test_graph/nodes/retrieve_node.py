from work_flow.test_graph.test_state import TestState



async def retrieve_node(state:TestState)->TestState:
    requirement_structured = state["requirement_structured"]
    return {}