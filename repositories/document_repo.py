from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Document, DocumentChunk


class DocumentRepo:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def add_document(self,document:Document)->Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_document_by_file_hash(self,file_hash:str)->Document | None:
        stmt = select(Document).where(Document.file_hash == file_hash).limit(1)

        return (await self.session.execute(stmt)).scalar_one_or_none()