from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel

from search.models import RetrieveChunk

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


@dataclass
class PlanReason(BaseModel):
    count:int

    reason:str



class CopyRagDocumentState(TypedDict,total=False):

    # 路由，查看是否需要重写，多路召回，ai假写，或者原始问题不用动
    question_route:question_choose

    # 意图识别，查看是否是闲聊，或者流程查询，还是规章制度查询
    intent:intent_choose

    # 历史对话
    history:list[Message]

    # 会话id
    conversation_id:UUID

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

    # 文档检索相关性打分
    relevance_score:float|None

    # 文档打分说明
    relevance_reason:str

    # 循环次数
    cycle_count:int

    # 循环说明，第几次，为什么重复循环
    plan_reasons:list[PlanReason]