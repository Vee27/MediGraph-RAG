from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    patient_id: str = ""


class SourceChunk(BaseModel):
    text: str
    page: int
    score: float


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str
    sources: list[SourceChunk] = []
    intent: str = ""


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str


class UploadResponse(BaseModel):
    message: str
    patient_id: str
    chunks_stored: int
    pages_processed: int