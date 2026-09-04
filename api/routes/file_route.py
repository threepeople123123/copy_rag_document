from pathlib import PurePath
from xml.dom import ValidationErr

from fastapi import APIRouter, UploadFile, File

from api.schemas.file import FileResponse
from db.session import DbSession
from stroe import file_service

route = APIRouter(prefix="/file", tags=["file"])

@route.post("/push/skill",status_code=201,response_model=FileResponse)
async def push_skill(session:DbSession,file:UploadFile=File(...,description="skill文件上传"))->FileResponse:
    suffix = PurePath(file.filename or "").suffix

    if suffix != ".zip":
        raise ValidationErr("请上传压缩包")
    service = file_service.FileService(session)
    await service.put_skill_by_zip(file)
    return FileResponse("xip")
