from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

event_type = Literal["data","error"]

@dataclass
class SseIterator:
    event:event_type
    data:dict


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # 注意:min_length/max_length 只能用于 str/list 类型;
    # UUID 类型本身就会校验格式,加长度约束会报
    # "Unable to apply constraint 'min_length' to supplied value ..."
    conversation_id: UUID
    # skill名称
    skill:str