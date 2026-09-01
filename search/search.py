import asyncio
from dataclasses import dataclass
from uuid import UUID

from core.config import settings
from core.logging import get_logger
from db.session import AsyncSessionLocal
from search.keyword_retrieve import KeyWordRetrieve
from search.vectroy_retrieve import VectorRetrieve
from work_flow.copy_rag_document_state import CopyRagDocumentState

logger = get_logger(__name__)
@dataclass
class RetrieveChunk:
    # 内容
    chunk_id: UUID
    document_id: UUID
    content:str
    vector_rank: int | None = None
    vector_score: float | None = None  # 原始 cosine similarity（向量路命中时填充）
    keyword_rank: int | None = None
    keyword_score: float | None = None  # 原始 ts_rank（关键词路命中时填充）
    rrf_score: float | None = None
    rerank_score: float | None = None


def rrf_rank(keyword:list[RetrieveChunk], vector:list[RetrieveChunk], top_k:int)->list[RetrieveChunk]:
    keyword_rank_map: dict[UUID, RetrieveChunk] = {
        chunk.chunk_id: chunk
        for chunk in keyword
    }
    rrf_result:list[RetrieveChunk] = []
    for v in vector:
        if v.vector_score >= (settings.vector_score or 0.0):
            keyword = keyword_rank_map.get(v.chunk_id)
            if keyword:
                rrf_score = (1 / (keyword.keyword_rank+60) + 1 / (v.vector_rank + 60))
                v.keyword_score = keyword.keyword_score
                v.keyword_rank = keyword.keyword_rank
                v.rrf_score = rrf_score
            rrf_result.append(v)
    return sorted(rrf_result,key=lambda rrf:rrf.rrf_score or 0.0, reverse=True)[:top_k]


class Search:

    async def _safe_search(self,search_cls: type[KeyWordRetrieve] | type[VectorRetrieve],question:str,top_k:int,label:str)->list[RetrieveChunk]:
        try:
            async with AsyncSessionLocal() as session:
                retriever = search_cls(session)
                result = retriever.search(question,top_k)
                return result
        except Exception as e:
            logger.error(f"查询{label}报错，报错原因：{e}")
            return []



    async def search_all(self,question:str,top_k)->list[RetrieveChunk]:

        keyword_result ,vector_result = await asyncio.gather(
            self._safe_search(KeyWordRetrieve,question,top_k,"keyword"),
            self._safe_search(VectorRetrieve, question, top_k,"vector"),
        )

        return rrf_rank(keyword_result,vector_result,top_k)


