from datetime import datetime
from enum import Enum
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import UUID, Text, DateTime, func, Computed, ForeignKey, String, BigInteger
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class DocumentStatus(str,Enum):
    """
    UPLOADING. 上传中
    PARSING  解析中
    INDEXING 切分 + 向量化 + 写chunk中
    READY    已经准备好
    FAILED   失败
    """
    UPLOADING = "uploading"
    PARSING = "parsing"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"

class Document(Base):
    __tablename__ = 'document'

    id:Mapped[UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("document.id",ondelete="CASCADE") ,primary_key=True,default=uuid4)
    title:Mapped[str] = mapped_column(String(255),nullable=False)
    file_hash:Mapped[str] = mapped_column(String(64),nullable=False)
    content_type:Mapped[str] = mapped_column(String(50),nullable=False)
    size:Mapped[int] = mapped_column(BigInteger,nullable=False)
    bucket_name:Mapped[str] = mapped_column(String(255),nullable=False)
    status:Mapped[DocumentStatus] = mapped_column(String(10),nullable=False,default=DocumentStatus.UPLOADING)
    error_message:Mapped[str | None] = mapped_column(Text,nullable=True)
    create_at :Mapped[datetime] = mapped_column(DateTime(timezone=True),default=func.now(),nullable=False)
    update_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now(),nullable=False)

    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")

class DocumentChunk(Base):
    __tablename__ = 'document_chunk'

    id:Mapped[UUID] = mapped_column(UUID(as_uuid=True) ,primary_key=True,default=uuid4)
    document_id:Mapped[UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("document.id",ondelete="CASCADE"),default=uuid4)
    vector:Mapped[list[float]] = mapped_column(Vector(1024),nullable=False)
    content:Mapped[str] = mapped_column(Text,nullable=False)
    chunk_metadata:Mapped[dict] = mapped_column(JSONB,nullable=False)
    create_at :Mapped[datetime] = mapped_column(DateTime(timezone=True),default=func.now(),nullable=False)
    content_parser:Mapped[str] = mapped_column(TSVECTOR,Computed("to_tsvector('chinese_zhparser' , content)",persisted=True),nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")

class File(Base):
    __tablename__ = "file"

    id:Mapped[UUID] = mapped_column(UUID(as_uuid=True),default=uuid4,primary_key=True)
    type:Mapped[str] = mapped_column(String(20),nullable=True)
    object_name:Mapped[str] = mapped_column(String(400),nullable=False)
    file_name:Mapped[str] =  mapped_column(String(50),nullable=False)
    create_at :Mapped[datetime] = mapped_column(DateTime(timezone=True),default=func.now(),nullable=False)