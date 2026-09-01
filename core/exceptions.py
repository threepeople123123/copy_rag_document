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