from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI

from apis import simiraity_api_router
from io_controller import IOController
from manager import ImageSearchManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any, None]:
    IOController.initialize()
    ImageSearchManager.initialize()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(simiraity_api_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
