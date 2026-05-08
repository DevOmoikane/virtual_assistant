from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from virtual_assistant_be.api.routes.ws import router as ws_router
from virtual_assistant_be.api.routes.health import router as health_router

import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(filename)s:%(funcName)s[%(lineno)d] - %(message)s",
    datefmt="[%Y/%m/%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)]
)
logging.getLogger("asyncio").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    app = FastAPI()

    # CORS settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (for development)
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
        allow_credentials=True,  # Allow cookies and credentials
    )

    app.include_router(ws_router)
    app.include_router(health_router)

    return app

app = create_app()
