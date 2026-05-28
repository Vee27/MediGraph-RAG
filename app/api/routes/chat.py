from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.models.response_models import (
    ChatRequest, ChatResponse, SourceChunk
)
from app.models.ollama_client import OllamaClient
from app.api.dependencies import get_ollama_client
from app.agents.graph_builder import run_rag_graph
from app.agents.summarizer_agent import run_soap_node
from app.memory.session_memory import session_memory
from app.config.settings import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical assistant helping physicians
understand patient charts. Answer accurately and concisely using
only verified information."""


class SOAPRequest(BaseModel):
    patient_id: str
    session_id: str = "default"


class SOAPResponse(BaseModel):
    soap_note: str
    patient_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client: OllamaClient = Depends(get_ollama_client),
):
    logger.info(
        f"Chat — session: {request.session_id} | "
        f"patient: '{request.patient_id}' | "
        f"msg: '{request.message[:60]}'"
    )

    try:
        if request.patient_id:
            # Get conversation history for this session
            history = session_memory.get_history(request.session_id)

            logger.info(
                f"Session {request.session_id} has "
                f"{len(history)} previous messages"
            )

            # Run the full RAG graph with history
            final_state = await run_rag_graph(
                message=request.message,
                patient_id=request.patient_id,
                session_id=request.session_id,
                history=history,
            )

            # Save this exchange to memory
            session_memory.add_message(
                session_id=request.session_id,
                role="user",
                content=request.message,
                patient_id=request.patient_id,
            )
            session_memory.add_message(
                session_id=request.session_id,
                role="assistant",
                content=final_state["reply"],
            )

            sources = [
                SourceChunk(
                    text=c.get("text", "")[:200],
                    page=c.get("page", 0),
                    score=float(
                        c.get("score", c.get("hybrid_score", 0.0))
                    ),
                )
                for c in final_state.get("sources", [])
            ]

            return ChatResponse(
                reply=final_state["reply"],
                session_id=request.session_id,
                model=settings.ollama_chat_model,
                sources=sources,
                intent=final_state.get("intent", ""),
            )

        else:
            # Direct Ollama — still use memory for continuity
            history = session_memory.get_history(request.session_id)

            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + history
                + [{"role": "user", "content": request.message}]
            )

            reply = await client.chat(messages=messages)

            # Save to memory
            session_memory.add_message(
                session_id=request.session_id,
                role="user",
                content=request.message,
            )
            session_memory.add_message(
                session_id=request.session_id,
                role="assistant",
                content=reply,
            )

            return ChatResponse(
                reply=reply,
                session_id=request.session_id,
                model=settings.ollama_chat_model,
                sources=[],
                intent="direct",
            )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/soap", response_model=SOAPResponse)
async def generate_soap(request: SOAPRequest):
    logger.info(f"SOAP — patient: {request.patient_id}")

    try:
        state = {
            "message":    "Generate SOAP note",
            "patient_id": request.patient_id,
            "session_id": request.session_id,
            "context":    "",
            "sources":    [],
            "reply":      "",
            "error":      "",
            "intent":     "soap",
            "history":    [],
        }

        final_state = await run_soap_node(state)

        return SOAPResponse(
            soap_note=final_state["reply"],
            patient_id=request.patient_id,
        )

    except Exception as e:
        logger.error(f"SOAP failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a session's conversation history.
    Call this when starting a new patient case.
    """
    session_memory.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.get("/session/{session_id}/stats")
async def session_stats(session_id: str):
    """Get stats about a conversation session."""
    return session_memory.get_session_stats(session_id)