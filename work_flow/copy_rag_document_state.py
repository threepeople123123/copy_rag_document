from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypedDict

from search.search import RetrieveChunk

intent_choose=Literal["small_talk","rules_regulations","work_flow"]

question_choose=Literal["multi_channel_recall","assistant_false_answer","original"]


class MessageRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"



@dataclass
class Message:

    role:MessageRole

    content:str



class CopyRagDocumentState(TypedDict,total=False):

    # 路由，查看是否需要重写，多路召回，ai假写，或者原始问题不用动
    question_route:question_choose

    # 意图识别，查看是否是闲聊，或者流程查询，还是规章制度查询
    intent:intent_choose

    # 历史对话
    history:list[Message]

    # 会话id
    conversation_id:str

    # 用户问题
    question:str

    # ai 回答
    assistant_answer:str

    # 问题重写
    question_rewrite:str

    # 问题重写
    multi_channel_recall: list[str]

    # ai虚假回答
    assistant_false_answer:str

    # 检索到的文档
    retrieve:list[RetrieveChunk]







