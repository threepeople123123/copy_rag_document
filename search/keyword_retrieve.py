from sqlalchemy.ext.asyncio import AsyncSession


class KeyWordRetrieve:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def search(self,question:str,top_k:int)->list[str]:
        ...
