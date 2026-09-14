import json

from llm.models import get_max_structured_agent
from llm.test_agent_prompt import build_finally_messages
from work_flow.test_graph.test_state import TestState, TestCaseDimensionWeb, TestCaseDimension


async def finally_node(state:TestState)->TestState:

    return_web_requirement_structured = state.get("return_web_requirement_structured",[])
    history = state.get("history",[])

    update:TestState = {}

    if return_web_requirement_structured:

        requirement_structured_list = [
            re
            for re in return_web_requirement_structured if re.supplement
        ]
        if len(requirement_structured_list) > 0:
            result_list:list[str] = []
            for index,requirement in enumerate(requirement_structured_list,start=1):
                result = parser_requirement(index,requirement)
                result_list.append(result)
            if len(result_list) > 0:
                messages = build_finally_messages(json.dumps(result_list,ensure_ascii=True,indent=4),history)
                max_structured_agent = get_max_structured_agent(list[TestCaseDimension])
                response:list[TestCaseDimension] = await max_structured_agent.ainvoke(messages)
                update["requirement_structured"] = response
    return update

def parser_requirement(index:int,requirement:TestCaseDimensionWeb)->str:
    supplement = requirement.requirement
    requirement.supplement = ""
    return f"""
        第{index}段：
        {requirement.model_dump_json(indent=4)}\n\n 用户提出需要修改的点：{supplement}
    """