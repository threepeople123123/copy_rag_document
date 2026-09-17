from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from llm.models import get_chat_model
from react_agent.nodes.prompt_node import _SYSTEM_MAIN_PROMPT
from react_agent.react_state import ReActState
from react_agent.tool.registry import get_tool_registry
from react_agent.tool.tools import EXIT_TOOL_NAME
from work_flow.schemas.graph_schemas import Message, MessageRole

MAX_ITERATIONS = 12

_SYSTEM_PROMPT = (
    _SYSTEM_MAIN_PROMPT
    + "\n\n# 结束方式（重要）\n"
    + "当你确认任务已经完成、可以交付最终结果时，必须调用 finish 工具，"
    + "把完整、可直接交付的最终答案放到 final_answer 参数里，然后结束。"
    + "只有调用 finish 才能结束流程。"
)


def history_to_message(history: list[Message]) -> list[BaseMessage]:
    """把自定义 Message 历史转成 LangChain 消息（对 role 同时兼容枚举与字符串）。"""
    messages: list[BaseMessage] = []
    for msg in history:
        role = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
        if role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif role == "system":
            messages.append(SystemMessage(content=msg.content))
        elif role == "tool":
            messages.append(ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id))
    return messages


_agent_llm = None


def get_agent_llm():
    """绑定所有工具后的 LLM（懒加载并缓存，避免每次循环重复 bind）。"""
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = get_chat_model().bind_tools(get_tool_registry().get_all())
    return _agent_llm


async def agent_node(state: ReActState) -> dict:
    messages = list(state.get("base_messages") or [])

    # 首次进入：系统提示 + 历史 + 当前问题
    if not messages:
        messages = [SystemMessage(content=_SYSTEM_PROMPT)]
        messages += history_to_message(state.get("history") or [])
        messages.append(HumanMessage(content=state["question"]))

    response = await get_agent_llm().ainvoke(messages)

    return {
        "base_messages": messages + [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def route_after_agent(state: ReActState) -> str:
    """agent 之后的路由：有工具调用→执行工具；调用了退出工具或直接给答案→结束。"""
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "finish"

    messages = state.get("base_messages") or []
    if not messages:
        return "finish"

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if any(tc.get("name") == EXIT_TOOL_NAME for tc in last.tool_calls):
            return "finish"
        return "tools"

    # 模型没有调用任何工具（直接输出文字）→ 也结束
    return "finish"
