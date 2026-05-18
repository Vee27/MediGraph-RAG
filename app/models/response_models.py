from pydantic import BaseModel


class ChatRequest(BaseModel):
    """What the user sends to /chat"""
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """What /chat sends back"""
    reply: str
    session_id: str
    model: str


class HealthResponse(BaseModel):
    """What /health sends back"""
    status: str
    ollama_reachable: bool
    model: str