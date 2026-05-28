import logging
from app.rag.prompts.soap_templates import build_soap_messages
from app.models.ollama_client import ollama_client
from app.rag.retrieval.hybrid_search import hybrid_search
from app.rag.retrieval.metadata_filter import format_context

logger = logging.getLogger(__name__)


async def run_soap_node(state: dict) -> dict:
    """
    LangGraph node — generates a SOAP note from chart context.

    Reads from state:  context, patient_id
    Writes to state:   reply

    If context is empty (SOAP called directly), fetches
    a broad set of chunks from ChromaDB first.
    """
    logger.info("[soap_node] Generating SOAP note")

    context = state.get("context", "")
    patient_id = state.get("patient_id", "")

    if not context:
        if not patient_id:
            return {
                **state,
                "reply": "No patient chart uploaded. Use /upload first.",
            }

        logger.info("[soap_node] Fetching broad context for SOAP")
        chunks = await hybrid_search(
            query="patient history diagnosis medications vitals symptoms plan",
            patient_id=patient_id,
            top_k=8,
        )
        context = format_context(chunks) if chunks else ""

    if not context:
        return {
            **state,
            "reply": "No chart data found for this patient.",
        }

    try:
        messages = build_soap_messages(context=context)
        soap_note = await ollama_client.chat(messages=messages, temperature=0.1)
        logger.info("[soap_node] SOAP note generated")
        return {**state, "context": context, "reply": soap_note}

    except Exception as e:
        logger.error(f"[soap_node] Failed: {e}")
        return {**state, "reply": f"SOAP generation failed: {str(e)}", "error": str(e)}