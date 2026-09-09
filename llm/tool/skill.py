from dataclasses import dataclass

from langchain.tools import  ToolRuntime

from db.session import AsyncSessionLocal
from stroe.file_service import FileService
from work_flow.rag_graph.copy_rag_document_state import RagDocumentState


@dataclass
class SkillMateData:
    skill_name:str
    description:str

async def skill_list(runtime:ToolRuntime[RagDocumentState])->list[SkillMateData]:
    """
    查找可用skill集合，会返回 SKILL信息对象， 包含名称(name)和描述(description)，供你选择
    """
    async with AsyncSessionLocal() as session:
        file_service = FileService(session)
        file_list = await file_service.find_skill(name="",description="")
        return [
            SkillMateData(skill_name=file.skill_name,description=file.skill_description)
            for file in file_list
        ]


async def find_skill(skill_name:str, description:str, runtime:ToolRuntime[RagDocumentState])->list[SkillMateData]:
    """查找指定skill，使用skill_name参数进行查找，我会返回 SKILL信息对象， 包含名称(name)和描述(description)，供你选择

    Args:
        skill_name:skill名称 （require=true）
        description: skill描述（require=true）
    """
    conversation_id = runtime.context.conversation_id
    async with AsyncSessionLocal() as session:
        file_service = FileService(session)
        file_list = await file_service.find_skill(name=skill_name,description=description)
        return [
            SkillMateData(skill_name=file.skill_name,description=file.skill_description)
            for file in file_list
        ]

