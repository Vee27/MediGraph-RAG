import re
import logging

logger = logging.getLogger(__name__)

DOSAGE_PATTERN = r"\b\d+\.?\d*\s*(?:mg|mcg|ml|g|units?)\b"
NUMBER_PATTERN = r"\b\d+\.?\d*\b"


def extract_claims(text: str) -> list[str]:
    """
    Split a reply into individual sentences for claim checking.
    Each sentence is treated as one claim.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def check_numbers_in_context(reply: str, context: str) -> list[str]:
    """
    Find numeric values in the reply that don't appear in context.
    Numbers are the most common source of hallucination in medical RAG.

    Returns:
        List of suspicious numeric values
    """
    reply_numbers = re.findall(NUMBER_PATTERN, reply)
    suspicious = []

    for num in reply_numbers:
        # Allow common non-clinical numbers (page refs, list numbers)
        if int(float(num)) < 3:
            continue
        if num not in context:
            suspicious.append(num)

    return suspicious


def check_dosages_in_context(reply: str, context: str) -> list[str]:
    """
    Find dosages in reply not present in context.
    e.g. "500mg" in reply but not in context → suspicious.

    Returns:
        List of unverified dosage strings
    """
    reply_dosages = re.findall(DOSAGE_PATTERN, reply, re.IGNORECASE)
    unverified = [
        d for d in reply_dosages
        if d.strip().lower() not in context.lower()
    ]
    return unverified


def run_hallucination_check(
    reply: str,
    context: str,
    session_id: str = "",
) -> dict:
    """
    Run all hallucination checks on a reply.

    This is a lightweight heuristic check — not ML-based.
    It looks for:
    - Dosages in reply not found in context
    - Numeric values in reply not found in context

    Results are logged but never shown to the user.
    The reply is never modified.

    Args:
        reply:      The LLM's answer
        context:    The retrieved chunks used to generate it
        session_id: For logging correlation

    Returns:
        Dict with check results for developer inspection:
        {
            "passed": True/False,
            "unverified_dosages": [...],
            "suspicious_numbers": [...],
            "reply_length": int,
            "context_length": int,
        }
    """
    if not reply or not context:
        return {
            "passed": True,
            "unverified_dosages": [],
            "suspicious_numbers": [],
            "reply_length": len(reply),
            "context_length": len(context),
        }

    unverified_dosages = check_dosages_in_context(reply, context)
    suspicious_numbers = check_numbers_in_context(reply, context)

    passed = (
        len(unverified_dosages) == 0
        and len(suspicious_numbers) == 0
    )

    if not passed:
        logger.warning(
            f"[hallucination_check] session={session_id} | "
            f"FAILED | "
            f"unverified_dosages={unverified_dosages} | "
            f"suspicious_numbers={suspicious_numbers}"
        )
    else:
        logger.info(
            f"[hallucination_check] session={session_id} | PASSED"
        )

    return {
        "passed": passed,
        "unverified_dosages": unverified_dosages,
        "suspicious_numbers": suspicious_numbers,
        "reply_length": len(reply),
        "context_length": len(context),
    }