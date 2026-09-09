from dataclasses import dataclass
from typing import Literal, TypedDict, dataclass_transform

from pydantic import Field, BaseModel

from work_flow.rag_graph.copy_rag_document_state import Message

intention_choose = Literal["requirement","small_talk","requirement_supplement"]


class DimensionItem(BaseModel):
    id:str = Field(...,description="id标识")

    description:str = Field(...,description="具体的描述")


class TestCaseDimension(BaseModel):

    requirement:str = Field(..., description="拆分的需求")

    steps:list[DimensionItem] = Field(default_factory=list, description="拆分的功能点的每个步骤，例如：第一步：打开oa系统页面，第二步：填报数据，第三步：进行提交")

    functionality:list[DimensionItem] = Field(default_factory=list, description="测试的功能")

    boundaries:list[DimensionItem] = Field(default_factory=list, description="测试的边界，从多个维度生成")

    exceptions:list[DimensionItem] = Field(default_factory=list, description="异常测试，从多个异常出发生成")

    security:list[DimensionItem] = Field(default_factory=list, description="测试的安全，多个方面，网络方面，字段输入过长亦或者数字输入负数")



class TestState(TypedDict, total=False):

    # 会话id
    conversation_id:str

    # 用户问题
    question:str

    # 用户意图识别
    intention:intention_choose

    # 需求结构化
    requirement_structured:list[TestCaseDimension]

    # 历史对话
    history:list[Message]








