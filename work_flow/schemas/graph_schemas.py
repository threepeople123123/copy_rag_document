from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Message:

    role:MessageRole

    content:str

    tool_call_id:str