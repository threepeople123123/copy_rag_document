import base64
from pathlib import PurePath
from xml.dom import ValidationErr

from fastapi import APIRouter, UploadFile, File

from api.schemas.file import FileResponse
from db.session import DbSession
from llm.models import get_chat_model_for_qwen3_8_max
from llm.prompts import build_skill_messages
from stroe import file_service

route = APIRouter(prefix="/file", tags=["file"])

@route.post("/push/skill",status_code=201,response_model=FileResponse)
async def push_skill(session:DbSession,file:UploadFile=File(...,description="skill文件上传"))->FileResponse:
    suffix = PurePath(file.filename or "").suffix
    if file.read() is None:
        raise ValidationErr("请上传文件")
    if suffix != ".zip":
        raise ValidationErr("请上传压缩包")
    service = file_service.FileService(session)
    result = await service.put_skill_by_zip(file)
    return FileResponse(object_name=result)


@route.post("/push/image",status_code=201,response_model=FileResponse)
async def push_images(question:str,file:UploadFile=File(...,description="图片上传"))->FileResponse:
    model = get_chat_model_for_qwen3_8_max()
    message = build_skill_messages(question,history=[],history_compress="暂无",file_type="image", mini_type=file.content_type, file_base64=base64.b64encode(await file.read()).decode("utf-8"))
    response = await model.ainvoke(message)
    print(response.content)
    return FileResponse(object_name=response.content)


