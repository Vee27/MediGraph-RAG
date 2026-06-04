import logging
from typing import TypedDict
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.metadata_filter import filter_by_patient, format_context
from app.rag.prompts.clinical_prompts import build_rag_messages
from app.models.ollama_client import ollama_client

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    message: str
    patient_id: str
    session_id: str
    context: str
    sources: list[dict]
    reply: str
    error: str
    intent: str
    history: list[dict]


async def retrieve_node(state: AgentState) -> AgentState:
    """Find relevant chunks from ChromaDB via hybrid search."""
    logger.info(f"[retrieve_node] query: '{state['message'][:60]}'")

    try:
        chunks = await hybrid_search(
            query=state["message"],
            patient_id=state["patient_id"],
            top_k=3,
        )
        chunks = filter_by_patient(chunks, state["patient_id"])

        if not chunks:
            return {
                **state,
                "context": "No relevant information found in this patient's chart.",
                "sources": [],
            }

        return {
            **state,
            "context": format_context(chunks),
            "sources": chunks,
        }

    except Exception as e:
        logger.error(f"[retrieve_node] Failed: {e}")
        return {
            **state,
            "context": "Error retrieving chart context.",
            "sources": [],
            "error": str(e),
        }


async def generate_node(state: AgentState) -> AgentState:
    """Generate a grounded answer using retrieved context including history
       for follow-up questions."""
    logger.info("[generate_node] Generating answer")

    try:
        # build RAG prompt with retrieved context + question
        rag_messages = build_rag_messages(
            context=state["context"],
            question=state["message"],
        )

        # If there is conversation history, inject it between
        # the system message and the current user message.
        # This gives the LLM full conversation context.
        history = state.get("history", [])

        if history:
            system_msg = rag_messages[0]
            current_user_msg = rag_messages[1]

            messages_with_history = (
                [system_msg]
                + history        # previous turns
                + [current_user_msg]  # current question
            )
        else:
            messages_with_history = rag_messages

        reply = await ollama_client.chat(
            messages=messages_with_history,
            temperature=0.1,
        )

        return {**state, "reply": reply}

    except Exception as e:
        logger.error(f"[generate_node] Failed: {e}")
        return {
            **state,
            "reply": "Unable to generate a response. Please try again.",
            "error": str(e),
        }