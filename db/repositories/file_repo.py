from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import File


class FileRepo():
    def __init__(self,session:AsyncSession):
        self.session = session

    async def add_file(self,file:File):
        self.session.add(file)
        await self.session.flush()

    async def add_all_file(self, file: list[File]):
        self.session.add_all(file)
        await self.session.flush()

    async def find_skill_list(self, name:str = "",desc:str = "")->list[File]:
        stmt = select(File).where(File.file_type=="skill")
        # 参数非空才追加对应过滤条件；name/desc 都为空时 = 查出全部
        if name:
            stmt = stmt.where(File.skill_name == name)
        if desc:
            stmt = stmt.where(File.skill_description == desc)

        return [row[0] for row in (await self.session.execute(stmt)).all()]




