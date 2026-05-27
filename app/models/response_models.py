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

class UploadResponse(BaseModel):
    """What /upload sends back after processing a PDF"""
    message: str
    patient_id: str
    chunks_stored: int
    pages_processed: int