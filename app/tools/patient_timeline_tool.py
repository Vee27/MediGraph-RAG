import re
import logging
from app.rag.prompts.soap_templates import build_timeline_messages
from app.models.ollama_client import ollama_client

logger = logging.getLogger(__name__)

DATE_PATTERNS = [
    # ISO format: 2024-01-15
    r"\b\d{4}-\d{2}-\d{2}\b",
    # US slash format: 01/15/2024
    r"\b\d{2}/\d{2}/\d{4}\b",
    # US dash format: 01-15-2024
    r"\b\d{2}-\d{2}-\d{4}\b",
    # Long month name: January 15, 2024
    r"\b(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b",
    # Short month name: Jan 15, 2024 or Jan. 15 2024
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+\d{1,2},?\s+\d{4}\b",
    # Year only with context: in 2024, since 2023
    r"\b(?:in|since|from|until|by)\s+\d{4}\b",
]


def extract_dates_from_text(text: str) -> list[str]:
    """
    Find all date strings in text using regex.

    Returns:
        List of unique date strings found
    """
    found = []
    for pattern in DATE_PATTERNS:
        found.extend(re.findall(pattern, text, re.IGNORECASE))

    seen = set()
    unique = []
    for d in found:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


async def run_timeline_tool(state: dict) -> dict:
    """
    LangGraph node — extracts a chronological patient timeline.

    Reads from state:  context
    Writes to state:   reply
    """
    logger.info("[timeline_tool] Extracting patient timeline")

    context = state.get("context", "")

    if not context:
        return {
            **state,
            "reply": "No chart context available. Please upload a patient chart first.",
        }

    dates_found = extract_dates_from_text(context)
    logger.info(
        f"[timeline_tool] Dates found: {dates_found if dates_found else 'none'}"
    )

    try:
        messages = build_timeline_messages(context=context)
        reply = await ollama_client.chat(messages=messages, temperature=0.1)
        logger.info("[timeline_tool] Timeline generated")
        return {**state, "reply": reply}

    except Exception as e:
        logger.error(f"[timeline_tool] Failed: {e}")
        return {**state, "reply": f"Timeline extraction failed: {str(e)}", "error": str(e)}