import asyncio
from uuid import UUID

from core.config import settings
from core.logging import get_logger
from db.session import AsyncSessionLocal
from search.keyword_retrieve import KeyWordRetrieve
from search.models import RetrieveChunk
from search.vectroy_retrieve import VectorRetrieve

logger = get_logger(__name__)


def rrf_rank(keyword:list[RetrieveChunk] | None, vector:list[RetrieveChunk] | None, top_k:int)->list[RetrieveChunk]:
    if vector is None:
        return []
    keyword_rank_map:dict[UUID, RetrieveChunk] ={}
    if keyword :
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
                result = await retriever.search(question,top_k)
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


