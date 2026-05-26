import logging
import asyncio
from app.models.ollama_client import ollama_client

logger = logging.getLogger(__name__)


async def embed_text(text: str) -> list[float]:
    """
    Turn a single piece of text into an embedding vector.

    Args:
        text: Any string — a chunk, a question, a sentence

    Returns:
        List of 768 floats representing the meaning of the text
    """
    vector = await ollama_client.embed(text)
    return vector


async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed every chunk in the list.
    Adds an 'embedding' key to each chunk dict.

    Args:
        chunks: Output from chunk_pages()

    Returns:
        Same chunks but each now has an 'embedding' field:
        {
            "text": "Patient has hypertension...",
            "patient_id": "patient-001",
            "page": 1,
            "chunk_index": 0,
            "embedding": [0.23, -0.81, 0.44, ...]   ← added
        }

    Why one at a time?
        Ollama's embedding endpoint takes one text at a time.
        We loop through all chunks and embed each one.
        This is slow for large documents — Day 6 will handle
        batch storage more efficiently.
    """
    logger.info(f"Embedding {len(chunks)} chunks...")

    embedded = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Embedding chunk {i + 1}/{len(chunks)}")

        vector = await embed_text(chunk["text"])

        embedded.append({
            **chunk,           # keep all existing fields
            "embedding": vector,
        })

    logger.info(f"Done embedding {len(embedded)} chunks")
    return embedded