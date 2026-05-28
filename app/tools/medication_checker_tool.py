import re
import logging
from app.rag.prompts.soap_templates import build_medication_messages
from app.models.ollama_client import ollama_client

logger = logging.getLogger(__name__)

MEDICATION_KEYWORDS = [
    "metformin", "insulin", "glipizide", "januvia", "ozempic",
    "amlodipine", "lisinopril", "atenolol", "losartan", "metoprolol",
    "atorvastatin", "simvastatin", "rosuvastatin",
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "naproxen",
    "warfarin", "heparin", "clopidogrel", "rivaroxaban", "apixaban",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "tablet", "capsule", "injection",
]

DOSAGE_PATTERN = r"\b\d+\.?\d*\s*(?:mg|mcg|ml|units?|g)\b"


def quick_medication_scan(text: str) -> dict:
    """
    Fast keyword scan to detect medication mentions.

    Returns:
        Dict with found_keywords, found_dosages, has_medications
    """
    text_lower = text.lower()
    found_keywords = [kw for kw in MEDICATION_KEYWORDS if kw in text_lower]
    found_dosages = re.findall(DOSAGE_PATTERN, text, re.IGNORECASE)

    return {
        "found_keywords": found_keywords,
        "found_dosages": found_dosages,
        "has_medications": bool(found_keywords or found_dosages),
    }


async def run_medication_tool(state: dict) -> dict:
    """
    LangGraph node — extracts a structured medication list.

    Reads from state:  context
    Writes to state:   reply
    """
    logger.info("[medication_tool] Extracting medications")

    context = state.get("context", "")

    if not context:
        return {
            **state,
            "reply": "No chart context available. Please upload a patient chart first.",
        }

    scan = quick_medication_scan(context)
    logger.info(
        f"[medication_tool] Keywords: {scan['found_keywords']} | "
        f"Dosages: {scan['found_dosages']}"
    )

    try:
        messages = build_medication_messages(context=context)
        reply = await ollama_client.chat(messages=messages, temperature=0.1)
        logger.info("[medication_tool] Medication list generated")
        return {**state, "reply": reply}

    except Exception as e:
        logger.error(f"[medication_tool] Failed: {e}")
        return {**state, "reply": f"Medication extraction failed: {str(e)}", "error": str(e)}