from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import health, chat, upload
from app.config.logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MediGraph-RAG starting up...")
    yield
    logger.info("MediGraph-RAG shutting down...")


app = FastAPI(
    title="MediGraph-RAG",
    description="Agentic RAG chatbot for medical charts",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["system"])
app.include_router(chat.router, tags=["chat"])
app.include_router(upload.router, tags=["upload"])