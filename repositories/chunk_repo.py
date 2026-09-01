from typing import Sequence

from sqlalchemy import select, func, desc, ColumnElement, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db.models import DocumentChunk, Document, DocumentStatus


class ChunkRepo:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def add_chunk(self,chunks:Sequence[DocumentChunk]):
        self.session.add_all(chunks)
        await self.session.flush()

    async def search_chunks_by_embedding(self,embedding:Sequence[float],top_k:int)->list[tuple[DocumentChunk,float]]:

        distance = DocumentChunk.vector.cosine_distance(embedding)
        stmt = (
            select(DocumentChunk, distance.label("distance"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == DocumentStatus.READY.value)
            .order_by(distance.asc())
            .limit(top_k)
            .options(selectinload(DocumentChunk.document))
        )

        result = (await self.session.execute(stmt)).all()

        return [
            (chunk,float(distance))
            for chunk,distance in result]

    async def search_chunks_by_keyword(self,question:str,top_k:int)->list[tuple[DocumentChunk,float]]:

        # 1. 定义分词向量和查询条件
        tsquery = func.plainto_tsquery('chinese_zhparser', question)

        # 2. 定义相关性得分表达式并起个别名（方便排序和读取）
        rank_expr = func.ts_rank(DocumentChunk.content_parser,tsquery)

        conditions: list[ColumnElement[bool]] = [
            Document.status == "ready",
            DocumentChunk.content_parser.op("@@")(tsquery),
        ]
        # 3. 编写 SQL 语句
        stmt = (
            select(DocumentChunk, rank_expr)
            .where(and_(*conditions))
            .order_by(desc(rank_expr))
            .limit(top_k)
        )

        # 执行查询（返回的是一个包含 (Article, rank_score) 的元组列表）
        results = (await self.session.execute(stmt)).all()
        return [(chunk,float(rank)) for chunk,rank in results]



