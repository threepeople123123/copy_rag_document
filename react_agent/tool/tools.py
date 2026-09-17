import json

from langchain_core.tools import tool

from work_flow.test_graph.nodes.finally_node import finally_node
from work_flow.test_graph.nodes.generate_test_node import generate_test_node
from work_flow.test_graph.nodes.intention_node import intention_node
from work_flow.test_graph.nodes.requirement_parser_node import requirement_parser_node
from work_flow.test_graph.test_state import TestCaseDimension, TestCaseDimensionWeb

# 退出工具名：LLM 调用它表示任务完成、主动结束
EXIT_TOOL_NAME = "finish"


@tool
async def analyze_intention(question: str) -> str:
    """识别用户测试需求的意图，返回 requirement / small_talk / requirement_supplement 之一。"""
    result = await intention_node({"question": question, "history": []})
    return result.get("intention", "")


@tool
async def parse_requirement(question: str) -> str:
    """把用户的测试需求拆分成多个独立需求点，返回 JSON 字符串数组。"""
    result = await requirement_parser_node({"question": question, "history": []})
    items = result.get("requirement_structured", [])
    return json.dumps([item.requirement for item in items], ensure_ascii=False)


@tool
async def generate_test_cases(question: str) -> str:
    """根据用户需求生成完整测试用例（步骤/功能/边界/异常/安全五个维度），返回 JSON 数组。"""
    parsed = await requirement_parser_node({"question": question, "history": []})
    result = await generate_test_node({
        "question": question,
        "history": [],
        "requirement_structured": parsed.get("requirement_structured", []),
    })
    items = result.get("requirement_structured", [])
    return json.dumps([item.model_dump() for item in items], ensure_ascii=False)


@tool
async def modify_test_cases(current_test_cases: str, supplement: str) -> str:
    """根据用户的补充/修改要求，修改已有测试用例并返回修改后的完整测试用例 JSON。
    current_test_cases 是现有测试用例的 JSON 数组；supplement 是用户提出的修改点。"""
    if isinstance(current_test_cases, str):
        items = [TestCaseDimension.model_validate(x) for x in json.loads(current_test_cases)]
    else:
        items = [TestCaseDimension.model_validate(x) for x in current_test_cases]

    web_items = [
        TestCaseDimensionWeb(**x.model_dump(), supplement=supplement)
        for x in items
    ]

    result = await finally_node({
        "return_web_requirement_structured": web_items,
        "history": [],
    })
    out = result.get("requirement_structured", [])
    return json.dumps([x.model_dump() for x in out], ensure_ascii=False)


@tool
def finish(final_answer: str) -> str:
    """任务已经完成、可以交付最终结果时，必须调用本工具结束流程。
    final_answer 是完整、可直接交付给用户的最终内容。"""
    return final_answer
