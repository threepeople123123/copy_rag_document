from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from work_flow.schemas.graph_schemas import Message


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class ToolCallResponse:
    id: str
    name: str
    response: str = ""
    ok: bool = True
    error: str | None = None


class ReActState(TypedDict, total=False):
    # 会话id
    conversation_id: str

    # 问题查询
    question: str

    # 历史消息（自定义 Message，用于持久化到 redis）
    history: list[Message]

    # agent 循环的 LangChain 消息
    base_messages: list[BaseMessage]

    # 工具调用
    tool_call: list[ToolCall]

    # 工具调用结果
    tool_call_response: list[ToolCallResponse]

    # 退出工具输出的最终答案
    final_answer: str

    # 是否已结束
    finished: bool

    # 循环次数（防死循环）
    iteration_count: int
