from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.exceptions import AppException
from core.logging import get_logger

logger = get_logger(__name__)



async def _unhandled_app_exception_handler(request:Request,exc:AppException) ->JSONResponse:
    logger.exception("unhandled exception at %s %s",request.method,request.url.path)
    return JSONResponse(
        status_code=exc.http_status,
        content={"code":exc.code,"message":exc.message}
    )

async def _unhandled_exception_handler(request:Request,exc:Exception) ->JSONResponse:
    logger.exception("unhandled exception at %s %s",request.method,request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code":"internal","message":"服务内部错误"}
    )


def register_error_handlers(app:FastAPI)->None:
    # 必须分开注册:同一个类注册两次会覆盖。Starlette 按异常 MRO 选最具体的 handler,
    # AppException 的实例会命中第一个,其余异常命中第二个。
    app.add_exception_handler(AppException, _unhandled_app_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)




