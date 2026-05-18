from fastapi import APIRouter
from app.models.response_models import HealthResponse
from app.models.ollama_client import ollama_client
from app.config.settings import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the app and Ollama are both running.
    Returns ollama_reachable: true/false so you know
    if the LLM is available before sending chat requests.
    """
    reachable = await ollama_client.health_check()

    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama_reachable=reachable,
        model=settings.ollama_chat_model,
    )