from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.sync import update

from db.models import Document, DocumentChunk, DocumentStatus


class DocumentRepo:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def add_document(self,document:Document)->Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def update_state(self,document:Document,state:DocumentStatus,error_message: str | None = None,):
        if document is None:
            return
        document.status = state
        # 仅在显式传入时覆盖；保留 None 语义供成功状态清空之前的错误信息
        if error_message is not None or state == DocumentStatus.FAILED:
            document.error_message = error_message

    async def get_document_by_file_hash(self,file_hash:str)->Document | None:
        stmt = select(Document).where(Document.file_hash == file_hash).limit(1)

        return (await self.session.execute(stmt)).scalar_one_or_none()