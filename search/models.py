from dataclasses import dataclass
from uuid import UUID


@dataclass
class RetrieveChunk:
    # 内容
    chunk_id: UUID
    document_id: UUID
    content: str
    vector_rank: int | None = None
    vector_score: float | None = None  # 原始 cosine similarity（向量路命中时填充）
    keyword_rank: int | None = None
    keyword_score: float | None = None  # 原始 ts_rank（关键词路命中时填充）
    rrf_score: float | None = None
    rerank_score: float | None = None
