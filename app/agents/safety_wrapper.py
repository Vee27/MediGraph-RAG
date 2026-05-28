import re
import logging

logger = logging.getLogger(__name__)

DOSAGE_PATTERN = r"\b\d+\.?\d*\s*(?:mg|mcg|ml|g|units?)\b"


def apply_safety_wrapper(reply: str, context: str) -> str:
    """
    Silent quality check — runs after every LLM reply.

    Checks if dosages in the reply exist in the source context.
    If a dosage appears in the reply but NOT in the context,
    it may be hallucinated — this gets logged for developer review
    but the reply is returned unchanged.

    Nothing is appended to the reply. The output the user
    sees is exactly what the LLM produced.

    Args:
        reply:   LLM output
        context: Retrieved chunks used to generate the reply

    Returns:
        The original reply, unmodified
    """
    if not reply or not context:
        return reply

    reply_dosages = re.findall(DOSAGE_PATTERN, reply, re.IGNORECASE)

    for dosage in reply_dosages:
        if dosage.strip().lower() not in context.lower():
            logger.warning(
                f"[safety_wrapper] Dosage in reply not found in context: '{dosage}' "
                f"— may be hallucinated. Review this response."
            )

    return reply