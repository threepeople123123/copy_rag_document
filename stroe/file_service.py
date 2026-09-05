import io
import zipfile
from pathlib import Path
from uuid import uuid4

import frontmatter
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from stroe.minio_client import get_minio_client


class FileService:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def put_skill_by_zip(self,file:UploadFile):

        prefix = str(uuid4())
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



                # 2. 读取文件内容（如果需要对文件内容进行处理或上传）
                with zf.open(info) as file_obj:
                    file_content = file_obj.read()
                    name = Path(info.filename).name
                    if name == "SKILL.md":
                        name,description = parser_md(file_content)

                    await get_minio_client().put_object(build_skill_object_name(relative_path,prefix),file_content)


def parser_md(file_content:bytes)->tuple[str,str]:
    content_str = file_content.decode("utf-8")
    post = frontmatter.loads(content_str)

    # 获取 name 和描述
    name = post.get("name")
    description = post.get("description")
    return str(name),str(description)
def build_skill_object_name(file_name:str,prefix:str):
    return  f"skill/{prefix}+{file_name}"