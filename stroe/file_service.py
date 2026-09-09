import io
import json
import zipfile
from pathlib import Path
from uuid import uuid4
from xml.dom import ValidationErr

import frontmatter
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from db.models import File
from db.repositories.file_repo import FileRepo
from stroe.minio_client import get_minio_client


class FileService:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def put_skill_by_zip(self,file:UploadFile):


        prefix = str(uuid4())

        files:list[File] = []
        with zipfile.ZipFile(io.BytesIO(await file.read()), 'r') as zf:
            # infolist() 可以获取文件的详细信息
            for info in zf.infolist():
                # info.filename 包含了完整的目录层级，例如：
                # "images/avatar/user1.png" 或 "docs/readme.md"
                relative_path = info.filename

                # 1. 判断是目录还是文件
                if info.is_dir():
                    print(f"[目录]: {relative_path}")
                    continue

                file_model:File = File()
                object_name = build_skill_object_name(relative_path, prefix)
                prefix_path = object_name.split("/")[:-1]
                file_model.skill_prefix_path = "/".join(prefix_path)

                # 2. 读取文件内容（如果需要对文件内容进行处理或上传）
                with zf.open(info) as file_obj:
                    file_content = file_obj.read()
                    name = Path(info.filename).name
                    file_model.file_name = name
                    file_model.object_name = object_name
                    if name == "SKILL.md":
                        name,description = parser_md(file_content)
                        file_model.skill_description = description
                        file_model.skill_name = name
                        file_model.file_type = "skill"

                    await get_minio_client().put_object(object_name,file_content)
                    files.append(file_model)
        if files:
            object_name_list = await self.add_all_file(files)
            return json.dumps(object_name_list,ensure_ascii=False,indent=4)
        return ""



    async def add_file(self,file:File)->str:
        file_repo = FileRepo(self.session)

        await file_repo.add_file(file)
        await self.session.commit()
        await self.session.refresh(file)
        return file.object_name

    async  def add_all_file(self,files:list[File])->list[str]:
        file_repo = FileRepo(self.session)

        await file_repo.add_all_file(files)
        await self.session.commit()
        # refresh 只接受单个 ORM 实例，不能传 list；commit 后属性默认过期，
        # 在 async 下访问过期属性会触发 MissingGreenlet，所以逐个刷新
        for file in files:
            await self.session.refresh(file)
        return [ file.object_name for file in files]

    async def find_skill(self,name:str,description:str)->list[File]:
        file_repo = FileRepo(self.session)
        return await file_repo.find_skill_list(name,description)



def parser_md(file_content:bytes)->tuple[str,str]:
    content_str = file_content.decode("utf-8")
    post = frontmatter.loads(content_str)

    # 获取 name 和描述
    name = post.get("name")
    description = post.get("description")
    return str(name),str(description)
def build_skill_object_name(file_name:str,prefix:str):
    return  f"skill/{prefix}/{file_name}"