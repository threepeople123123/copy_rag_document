from http import HTTPStatus


class AppException(Exception):
    code: str = "internal_error"
    message: str = "服务器内部错误"
    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str | None = None, *, code: str | None = None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        super().__init__(self.message)


class UnsupportedMediaTypeError(AppException):
    """文件类型不被支持（415）。"""
    code: str = "unsupported_media_type"
    message: str = "不支持的文件类型"
    http_status: int = HTTPStatus.UNSUPPORTED_MEDIA_TYPE


class ParseError(AppException):
    """文档解析失败（422）。"""
    code: str = "parse_error"
    message: str = "文档解析失败"
    http_status: int = HTTPStatus.UNPROCESSABLE_ENTITY