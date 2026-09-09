from llm.models import get_max_structured_agent
from llm.test_agent_prompt import build_test_case_dimension_prompt_messages
from work_flow.test_graph.test_state import TestState, TestCaseDimension


async def generate_test_node(state: TestState) -> TestState:
    history = state.get("history", [])
    requirement_structured = state["requirement_structured"]

    structured_llm = get_max_structured_agent(TestCaseDimension)

    test_case_dimension_list: list[TestCaseDimension] = []
    # 逐个需求生成测试维度：单次调用的输入更小、目标更单一，结构化输出更稳定，准确率更高
    for requirement in requirement_structured:
        messages = build_test_case_dimension_prompt_messages(requirement.requirement, history=history)
        test_case_dimension: TestCaseDimension = await structured_llm.ainvoke(messages)

        # 兜底：requirement 必须与上游拆分结果保持一致，防止模型改写或丢失原始需求
        test_case_dimension.requirement = requirement.requirement
        test_case_dimension_list.append(test_case_dimension)
    return {"requirement_structured": test_case_dimension_list}
