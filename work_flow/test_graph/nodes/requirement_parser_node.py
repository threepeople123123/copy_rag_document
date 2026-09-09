from pydantic import BaseModel, Field

from llm.models import get_structured_agent
from llm.test_agent_prompt import build_requirement_parser_prompt_message
from work_flow.test_graph.test_state import TestState, TestCaseDimension


class RequirementParser(BaseModel):
    requirement:list[str] = Field(...,description="需求拆分后的输出")


async def requirement_parser_node(state:TestState)->TestState:
    question = state["question"]
    history = state.get("history",[])
    messages = build_requirement_parser_prompt_message(question,history)
    structured_agent = get_structured_agent(RequirementParser)

    result = await structured_agent.ainvoke({
        "messages": messages
    })

    structured_response:RequirementParser = result["structured_response"]
    requirement_list =[
        TestCaseDimension(requirement=n)
        for n in structured_response.requirement
    ]
    return {"requirement_structured":requirement_list}
