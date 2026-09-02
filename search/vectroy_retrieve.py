from sqlalchemy.ext.asyncio import AsyncSession

from llm.models import get_embeddings
from repositories.chunk_repo import ChunkRepo
from search.models import RetrieveChunk


class VectorRetrieve:

    def __init__(self,session:AsyncSession):
        self.session = session

    async def search(self,question:str,top_k:int)->list[RetrieveChunk]:
        embedding_model = get_embeddings()
        embedding = embedding_model.embed_query(question)

        chunk_repo = ChunkRepo(self.session)

        chunks = await chunk_repo.search_chunks_by_embedding(embedding,top_k)

        return [
            RetrieveChunk(
                chunk_id=chunk.id
                ,document_id=chunk.document_id
                ,content=chunk.content
                ,vector_rank=rank
                ,vector_score=1.0 - distance
            )
            for rank, (chunk ,distance) in enumerate(chunks,start=1)
        ]

