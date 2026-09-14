from typing import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse
from langgraph.types import Command
from pydantic import BaseModel

from work_flow.test_graph.test_case_graph import get_test_graph
from work_flow.test_graph.test_state import TestCaseDimensionWeb

route = APIRouter(prefix="/test_case", tags=["test_case"])


class ResumeRequest(BaseModel):
    """前端审阅/修改后的测试用例列表，作为 resume 值传回。"""
    reviewed: list[dict]


@route.post("/generate", status_code=200, response_class=EventSourceResponse)
async def generate_test_graph(question: str, conversation_id: str) -> AsyncIterable[dict]:
    config = {"configurable": {"thread_id": conversation_id}}
    init = {"question": question, "conversation_id": conversation_id}

    # 首次运行：跑到 human_node 的 interrupt 暂停；updates 里会带 __interrupt__
    async for result in get_test_graph().astream(input=init, config=config, stream_mode="updates"):
        if result:
            yield result


@route.post("/resume", status_code=200, response_class=EventSourceResponse)
async def resume_generate_test_graph(conversation_id: str, payload: list[TestCaseDimensionWeb]) -> AsyncIterable[dict]:
    config = {"configurable": {"thread_id": conversation_id}}

    # 恢复：Command(resume=...) 作为 input 传入，不能再带 input=init
    # resume 的值会被 human_node 里的 interrupt() 接收，节点用它更新 state
    async for result in get_test_graph().astream(Command(resume=payload, update={"return_web_requirement_structured": payload}), config=config, stream_mode="updates"):
        if result:
            yield result
