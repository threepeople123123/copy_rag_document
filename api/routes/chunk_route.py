from fastapi import APIRouter, File, UploadFile

from api.schemas.chunk import ChunkResponse
from db.session import DbSession
from service.chunk_service import ChunkService

route = APIRouter(prefix="/chunk",tags=["Chunk"])


@route.post("",response_model=ChunkResponse,status_code=201,operation_id="chunkDocument")
async def chunk_document(session:DbSession,file:UploadFile=File(...,description="需要解析的文档，支持 pdf,html,word,markdown 格式的文件"))->ChunkResponse:
    chunk_service = ChunkService(session)
    document = await chunk_service.chunk(file)
    return ChunkResponse(
        id=document.id,
        title=document.title,
        error_message=None
    )

