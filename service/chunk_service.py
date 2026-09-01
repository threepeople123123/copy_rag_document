import hashlib
from pathlib import PurePath, Path
from tempfile import NamedTemporaryFile
from xml.dom import ValidationErr

from docling.document_converter import DocumentConverter
from fastapi import UploadFile
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import DocumentChunk, Document
from llm.models import get_embeddings
from repositories.chunk_repo import ChunkRepo
from repositories.document_repo import DocumentRepo
from splitter.splitter_document import split



ALLOWED_SUFFIX = {
    "application/pdf" : ".pdf",                              # pdf
    "text/html" : ".html",                                    # html
    "application/msword": ".doc",                           # doc
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",  # docx
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/plain": ".md",  # md（部分浏览器报 text/plain）
}

logger = get_logger(__name__)

class ChunkService:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def chunk(self,file:UploadFile)->Document:
        suffix = PurePath(file.filename or "").suffix
        content_type = file.content_type
        if (suffix not in ALLOWED_SUFFIX.values()) or (content_type not in ALLOWED_SUFFIX.keys()) :
            raise ValidationErr(f"当前不支持：{suffix} 类型的文件")

        # 1. 创建临时文件
        with NamedTemporaryFile(
                suffix=suffix,
                delete=False
        ) as temp_file:

            # 2. 读取 UploadFile
            content = await file.read()

            # 3. 写入临时文件
            temp_file.write(content)
            temp_file.flush()

            temp_path = Path(temp_file.name)

        try:
            # 4. 创建 Docling Converter
            converter = DocumentConverter()

            # 5. 转换文件
            result = converter.convert(temp_path)

            # 6. 转换成 Markdown
            markdown = result.document.export_to_markdown()

            content = await file.read()

            file_hash = hashlib.sha256(content).hexdigest()

            document = Document(
                title=file.filename,
                file_hash=file_hash,
                content_type=content_type,
                size=len(content),
            )

            # 插入document文档
            document_repo = DocumentRepo(session=self.session)
            await document_repo.add_document(document)
            await self.session.commit()
            await self.session.refresh(document)

            document_list = await split(markdown, file.filename)

            embeddings = await get_embeddings().aembed_documents(
                texts = [
                    chunk.page_content for chunk in document_list
                ]
            )

            document_chunks = [
                DocumentChunk(
                    document_id=document.id,
                    vector=vec,
                    content=c.page_content,
                    chunk_metadata=c.metadata
                )
                for c, vec in zip(document_list, embeddings, strict=True)
            ]

            chunk_repo = ChunkRepo(self.session)
            await chunk_repo.add_chunk(document_chunks)
            await self.session.commit()
            await self.session.refresh(document)

            return document
        except Exception as e:
            logger.error(f"解析失败{e}")
            raise ValidationErr(f"解析失败{e}")

        finally:
            # 7. 删除临时文件
            temp_path.unlink(missing_ok=True)




