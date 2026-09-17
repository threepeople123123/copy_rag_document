from langchain_core.messages import AIMessage

from react_agent.react_state import ReActState
from react_agent.tool.tools import EXIT_TOOL_NAME


async def finish_node(state: ReActState) -> dict:
    """从退出工具调用（或最后的文字回复）里取出最终答案。"""
    messages = state.get("base_messages") or []
    last = messages[-1] if messages else None

    final_answer = ""
    if isinstance(last, AIMessage):
        if last.tool_calls:
            for tc in last.tool_calls:
                if tc.get("name") == EXIT_TOOL_NAME:
                    args = tc.get("args") or {}
                    final_answer = args.get("final_answer") or ""
                    break
        if not final_answer:
            content = last.content
            final_answer = content if isinstance(content, str) else str(content or "")

    return {"final_answer": final_answer, "finished": True}
