from app.models.ollama_client import OllamaClient, ollama_client


def get_ollama_client() -> OllamaClient:
    """
    FastAPI dependency — injects the shared OllamaClient instance
    into any route that needs it.

    Usage in a route:
        from fastapi import Depends
        from app.api.dependencies import get_ollama_client

        @router.post("/something")
        async def my_route(client: OllamaClient = Depends(get_ollama_client)):
            reply = await client.chat(...)
    """
    return ollama_client