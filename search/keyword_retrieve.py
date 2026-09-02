from sqlalchemy.ext.asyncio import AsyncSession

from repositories.chunk_repo import ChunkRepo
from search.models import RetrieveChunk


class KeyWordRetrieve:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def search(self,question:str,top_k:int)->list[RetrieveChunk]:

        chunk_repo = ChunkRepo(session=self.session)
        raw = await chunk_repo.search_chunks_by_keyword(question,top_k)

        return [
            RetrieveChunk(
                chunk_id=chunk.id
                ,content=chunk.content
                ,document_id=chunk.document_id
                ,keyword_rank=rank
                ,keyword_score=score
            )
            for rank, (chunk ,score) in enumerate(raw,start=1)
        ]

