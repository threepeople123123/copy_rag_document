import hashlib
from asyncio import to_thread
from pathlib import PurePath, Path
from tempfile import NamedTemporaryFile

from docling.document_converter import DocumentConverter
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.exceptions import AppException, ParseError, UnsupportedMediaTypeError
from core.logging import get_logger
from db.models import DocumentChunk, Document, DocumentStatus
from db.repositories.chunk_repo import ChunkRepo
from llm.models import get_embeddings
from db.repositories.document_repo import DocumentRepo
from service.doc_converter import convert_doc_to_docx
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
            raise UnsupportedMediaTypeError(message=f"当前不支持：{suffix} 类型的文件")

        content = await file.read()
        # content 在写入临时文件前已读过一次,这里不能再读(流已耗尽)
        file_hash = hashlib.sha256(content).hexdigest()

        document_repo = DocumentRepo(session=self.session)
        document_exist = await document_repo.get_document_by_file_hash(file_hash)
        if document_exist:
            raise AppException(message=f"已经存在相同的文件,{file.filename},请换文件重新上传")

        #  创建临时文件
        with NamedTemporaryFile(
                suffix=suffix,
                delete=False
        ) as temp_file:
            # 3. 写入临时文件
            temp_file.write(content)
            temp_file.flush()

            temp_path = Path(temp_file.name)

        parse_path = temp_path
        try:
            # docling 无法直接解析二进制 .doc,需先转成 .docx(优先 Word COM,其次 LibreOffice)
            if suffix == ".doc":
                parse_path = await to_thread(convert_doc_to_docx, temp_path)

            #  创建 Docling Converter
            converter = DocumentConverter()

            #  转换文件
            result = converter.convert(parse_path)
            #  转换成 Markdown
            markdown = result.document.export_to_markdown()
            document = Document(
                title=file.filename,
                file_hash=file_hash,
                content_type=content_type,
                size=len(content),
                bucket_name=settings.bucket_name,
            )

            # 插入document文档
            await document_repo.add_document(document)

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
            await document_repo.update_state(document,DocumentStatus.READY,None)

            # 文档 + 所有 chunk 一次性提交;任一步失败都会整体回滚,避免留下孤儿文档
            await self.session.commit()
            await self.session.refresh(document)

            return document
        except AppException:
            # 业务异常(如 .doc 转换失败、文件重复等)原样抛出,保留明确错误信息
            await self.session.rollback()
            raise
        except Exception as e:
            await self.session.rollback()
            logger.exception("解析失败: %s", file.filename)
            raise ParseError(message=f"解析失败:{(str(e)[:300])}")

        finally:
            # 7. 删除临时文件(.doc 转换出的 .docx 一并清理)
            temp_path.unlink(missing_ok=True)
            if parse_path != temp_path:
                parse_path.unlink(missing_ok=True)
