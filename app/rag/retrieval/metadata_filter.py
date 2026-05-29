import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_by_patient(
    chunks: list[dict],
    patient_id: str,
) -> list[dict]:
    """
    Keep only chunks that belong to a specific patient.

    This is a safety check — ChromaDB already filters by
    patient collection, but this adds an extra layer to
    make sure no cross-patient data leaks through.

    Args:
        chunks:     List of retrieved chunks
        patient_id: The patient we want

    Returns:
        Filtered list containing only this patient's chunks
    """
    filtered = [
        c for c in chunks
        if c.get("patient_id") == patient_id
    ]

    if len(filtered) != len(chunks):
        logger.warning(
            f"Filtered out {len(chunks) - len(filtered)} "
            f"chunks that didn't belong to patient: {patient_id}"
        )

    return filtered


def filter_by_page(
    chunks: list[dict],
    pages: list[int],
) -> list[dict]:
    """
    Keep only chunks from specific pages.
    Useful when the user asks about a specific section.

    Args:
        chunks: List of retrieved chunks
        pages:  Page numbers to keep e.g. [1, 2]

    Returns:
        Chunks from the specified pages only
    """
    return [c for c in chunks if c.get("page") in pages]


def filter_by_score(
    chunks: list[dict],
    min_score: float = 0.5,
    score_key: str = "score",
) -> list[dict]:
    """
    Remove chunks below a minimum relevance score.
    Prevents low-quality matches from polluting the context.

    Args:
        chunks:    List of retrieved chunks
        min_score: Minimum score to keep (0.0 to 1.0)
        score_key: Which score field to use

    Returns:
        Only chunks above the score threshold
    """
    filtered = [
        c for c in chunks
        if c.get(score_key, 0) >= min_score
    ]

    logger.info(
        f"Score filter ({score_key} >= {min_score}): "
        f"{len(chunks)} → {len(filtered)} chunks"
    )

    return filtered


def format_context(chunks: list[dict]) -> str:
    """
    Turn a list of chunks into a single context string
    that gets injected into the LLM prompt.

    Args:
        chunks: Retrieved and filtered chunks

    Returns:
        Formatted string like:
        [Source: Page 1]
        Patient: John Doe...

        [Source: Page 1]
        Medications: Metformin 500mg...
    """
    if not chunks:
        return "No relevant context found in the patient chart."

    parts = []
    for chunk in chunks:
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        parts.append(f"[Source: Page {page}]\n{text}")

    return "\n\n".join(parts)