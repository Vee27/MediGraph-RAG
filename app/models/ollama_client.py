import httpx
import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Async HTTP client that talks to your local Ollama server.
    Used by agents and the RAG pipeline throughout the project.
    """

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.chat_model = settings.ollama_chat_model
        self.embed_model = settings.ollama_embed_model

    # ------------------------------------------------------------------
    # CHAT
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a list of messages to Ollama and get a reply back.

        Args:
            messages:    List of {"role": "user"/"assistant"/"system", "content": "..."}
            model:       Override the default chat model if needed
            temperature: 0.0 = very focused, 1.0 = more creative

        Returns:
            The assistant's reply as a plain string
        """
        model = model or self.chat_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        logger.info(f"Sending chat request → model: {model}, messages: {len(messages)}")

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                reply = data["message"]["content"]
                logger.info(f"Received reply ({len(reply)} chars)")
                return reply

            except httpx.TimeoutException:
                logger.error("Ollama chat request timed out after 120s")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama returned HTTP error: {e.response.status_code}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during chat: {e}")
                raise

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """
        Turn a piece of text into an embedding vector.
        Used by the RAG pipeline from Day 5 onwards.

        Args:
            text: The text you want to embed (one chunk at a time)

        Returns:
            A list of floats — the embedding vector
        """
        payload = {
            "model": self.embed_model,
            "prompt": text,
        }

        logger.info(f"Embedding text ({len(text)} chars) with model: {self.embed_model}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                )
                response.raise_for_status()
                vector = response.json()["embedding"]
                logger.info(f"Embedding complete — vector length: {len(vector)}")
                return vector

            except httpx.TimeoutException:
                logger.error("Ollama embedding request timed out")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama embedding HTTP error: {e.response.status_code}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during embedding: {e}")
                raise

    # ------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Check if Ollama server is reachable.
        Called by the /health route to report ollama_reachable.

        Returns:
            True if Ollama is up, False if not
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


# Single shared instance — imported everywhere in the project
ollama_client = OllamaClient()