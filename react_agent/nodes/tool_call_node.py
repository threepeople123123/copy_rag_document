from langchain_core.messages import ToolMessage

from react_agent.react_state import ReActState, ToolCall
from react_agent.tool.tool_manage import tool_manage


async def tool_call_node(state: ReActState) -> dict:
    """执行上一条 AIMessage 里的 tool_calls，把结果作为 ToolMessage 追加回消息。"""
    messages = list(state.get("base_messages") or [])
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    calls = [
        ToolCall(
            id=tc.get("id", ""),
            name=tc["name"],
            args=tc.get("args") or {},
        )
        for tc in tool_calls
    ]

    responses = await tool_manage(calls)

    tool_messages = [
        ToolMessage(
            content=(r.response if r.ok else f"工具执行失败：{r.error}"),
            tool_call_id=r.id,
            name=r.name,
        )
        for r in responses
    ]

    return {
        "base_messages": messages + tool_messages,
        "tool_call": state.get("tool_call", []) + calls,
        "tool_call_response": state.get("tool_call_response", []) + responses,
    }
