CLINICAL_SYSTEM_PROMPT = """You are a clinical assistant helping \
physicians understand patient medical charts.

IMPORTANT RULES:
- Answer ONLY using the context provided below
- If the answer is not in the context, say "I cannot find that \
information in this patient's chart"
- Never invent medical details, dosages, or diagnoses
- Be concise and precise — use clinical language
- Always mention which page the information comes from if possible
- If asked about medications, always include dosage if available
"""


CLINICAL_RAG_TEMPLATE = """PATIENT CHART CONTEXT:
{context}

---

PHYSICIAN QUESTION:
{question}

---

Based strictly on the patient chart context above, please answer \
the physician's question. If the information is not in the context, \
say so clearly.
"""


def build_rag_messages(context: str, question: str) -> list[dict]:
    """
    Build the messages list to send to Ollama for a RAG query.

    This combines the system prompt (how to behave) with
    the RAG template (context + question) into the format
    Ollama expects.

    Args:
        context:  The retrieved chunks formatted as a string
        question: The user's question

    Returns:
        List of message dicts ready for ollama_client.chat()
    """
    user_content = CLINICAL_RAG_TEMPLATE.format(
        context=context,
        question=question,
    )

    return [
        {
            "role": "system",
            "content": CLINICAL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]