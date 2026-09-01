from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Document


class DocumentRepo:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def add_document(self,document:Document)->Document:
        self.session.add(document)
        await self.session.flush()
        return document
