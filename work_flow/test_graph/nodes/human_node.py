from langgraph.types import interrupt

from work_flow.test_graph.test_state import TestState, TestCaseDimensionWeb


async def human_node(state: TestState):
    requirement_structured = state.get("requirement_structured") or []

    # 快速拷贝：TestCaseDimension 逐个转成 TestCaseDimensionWeb（额外字段走默认值）
    web_list = [
        TestCaseDimensionWeb(**item.model_dump())
        for item in requirement_structured
    ]

    # 暂停并把 web_list 交给前端审阅；
    # 恢复时 interrupt() 会返回前端通过 Command(resume=...) 传回来的数据
    reviewed = interrupt(web_list)

    # 前端传回的通常是 JSON 反序列化后的 dict 列表，转回 TestCaseDimensionWeb 再写回 state
    reviewed_web = [
        item if isinstance(item, TestCaseDimensionWeb) else TestCaseDimensionWeb.model_validate(item)
        for item in (reviewed or [])
    ]

    return {"return_web_requirement_structured": reviewed_web}
