import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.routes import chunk_route, chat_route, file_route
from core.config import settings
from core.logging import configure_logging, get_logger
from handler.error_handler import register_error_handlers


def create_app()-> FastAPI:
    configure_logging()
    logger = get_logger(__name__)
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    logger.info("app start complete")

    app.include_router(chunk_route.route,prefix="/api")
    app.include_router(chat_route.route,prefix="/api")
    app.include_router(file_route.route,prefix="/api")


    return app

app = create_app()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)



