from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse

from core.logging import get_logger

logger = get_logger(__name__)

async def _unhandled_exception_handler(request:Request,exc:Exception) ->JSONResponse:
    logger.exception("unhandled exception at %s %s",request.method,request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code":"internal","message":"服务内部错误"}
    )


def register_error_handlers(app:FastAPI)->None:
    app.add_exception_handler(Exception,_unhandled_exception_handler)