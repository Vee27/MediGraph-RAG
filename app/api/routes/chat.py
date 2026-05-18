from fastapi import APIRouter, HTTPException, Depends
from app.models.response_models import ChatRequest, ChatResponse
from app.models.ollama_client import OllamaClient
from app.api.dependencies import get_ollama_client
from app.config.settings import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# This is the system prompt that shapes how the LLM behaves.
# It is sent with every single request — the model always
# "remembers" it is a clinical assistant.
SYSTEM_PROMPT = """You are a clinical assistant helping physicians 
understand patient charts. Answer questions accurately and concisely. 
If you are unsure about something, say so clearly. 
Never invent medical details that are not in the provided context."""


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client: OllamaClient = Depends(get_ollama_client),
):
    """
    Send a message and get a clinical assistant reply.

    Body:
        message:    Your question or instruction
        session_id: Identifies the conversation (used for memory later)

    Returns:
        reply:      The LLM's response
        session_id: Echoed back so the frontend can track it
        model:      Which Ollama model answered
    """
    logger.info(f"Chat request — session: {request.session_id}, message: {request.message[:60]}")

    try:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": request.message,
            },
        ]

        reply = await client.chat(messages=messages)

        return ChatResponse(
            reply=reply,
            session_id=request.session_id,
            model=settings.ollama_chat_model,
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {str(e)}"
        )