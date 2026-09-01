from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DocumentChunk


class ChunkRepo:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def add_chunk(self,chunks:Sequence[DocumentChunk]):
        self.session.add_all(chunks)
        await self.session.flush()
