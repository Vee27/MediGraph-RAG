import json
import logging
from app.models.ollama_client import ollama_client
from app.agents.retrieval_agent import AgentState

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# INTENT CLASSIFICATION PROMPT
# ----------------------------------------------------------------
# We give the LLM exactly four options and ask for JSON.
# Strict output format means we can parse it reliably.
# Few-shot examples teach the LLM what each intent looks like
# without it needing to guess.

INTENT_SYSTEM_PROMPT = """You are a clinical query classifier.
Your only job is to read a physician's question and classify it 
into exactly one of these four intents:

1. retrieve   — A specific factual question about the patient.
                Looking for a particular piece of information
                from the chart.
                Examples:
                - "What is the patient's blood pressure?"
                - "What did the ECG show?"
                - "Are there any allergies documented?"
                - "What were the sodium levels?"

2. medication — Any question about drugs, prescriptions, dosages,
                or medication history.
                Examples:
                - "What is she on?"
                - "Can you pull up her meds?"
                - "What's been prescribed?"
                - "Any drug interactions I should know about?"
                - "Is she taking anything for blood pressure?"

3. timeline   — Questions about sequence, history, progression,
                or when events occurred.
                Examples:
                - "Walk me through what happened"
                - "When was he first admitted?"
                - "How has the condition progressed?"
                - "What's the sequence of events?"
                - "Give me the history"

4. soap       — Requests for a summary, note, documentation,
                or structured write-up of the case.
                Examples:
                - "Draft the clinical note"
                - "Give me a summary of this case"
                - "Document this encounter"
                - "Write this up"
                - "Summarise the patient"

Respond ONLY with a JSON object in this exact format:
{"intent": "<one of: retrieve, medication, timeline, soap>", "reason": "<one sentence>"}

No other text. No explanation outside the JSON."""


INTENT_USER_TEMPLATE = """Classify this clinical query:

"{message}"

Respond with JSON only."""


# ----------------------------------------------------------------
# FALLBACK — keyword hints used only if LLM call fails
# ----------------------------------------------------------------
# This is now a safety net, not the primary mechanism.

FALLBACK_KEYWORDS = {
    "soap": ["soap", "summarise", "summarize", "summary", "write up",
             "clinical note", "document", "report"],
    "timeline": ["timeline", "history", "chronolog", "when did",
                 "what happened", "progression", "sequence"],
    "medication": ["medication", "medicine", "drug", "prescribed",
                   "dosage", "meds", "pharmacy", "tablet", "pill"],
}


def _keyword_fallback(message: str) -> str:
    """
    Last resort keyword check used only when Ollama is unreachable.
    Returns 'retrieve' as default if no keywords match.
    """
    msg = message.lower()
    for intent, keywords in FALLBACK_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            logger.warning(f"[router] Fallback keyword match → {intent}")
            return intent
    return "retrieve"


# ----------------------------------------------------------------
# LLM INTENT CLASSIFIER
# ----------------------------------------------------------------

async def classify_intent(message: str) -> tuple[str, str]:
    """
    Use phi3:mini to classify the user's message into one of
    four intents: retrieve, medication, timeline, soap.

    The LLM is asked to respond in strict JSON so we can
    parse it reliably. If parsing fails or Ollama is down,
    we fall back to keyword matching.

    Args:
        message: The user's raw question

    Returns:
        Tuple of (intent, reason)
        intent: one of retrieve/medication/timeline/soap
        reason: LLM's one-sentence explanation (for logging)
    """
    logger.info(f"[router] Classifying: '{message[:80]}'")

    messages = [
        {
            "role": "system",
            "content": INTENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": INTENT_USER_TEMPLATE.format(message=message),
        },
    ]

    try:
        raw = await ollama_client.chat(
            messages=messages,
            temperature=0.0,   # zero temperature = deterministic classification
        )

        # Strip any accidental markdown fences the LLM may add
        clean = raw.strip().strip("```json").strip("```").strip()

        parsed = json.loads(clean)
        intent = parsed.get("intent", "retrieve").strip().lower()
        reason = parsed.get("reason", "")

        # Validate — must be one of the four known intents
        valid = {"retrieve", "medication", "timeline", "soap"}
        if intent not in valid:
            logger.warning(
                f"[router] LLM returned unknown intent '{intent}' — "
                f"defaulting to retrieve"
            )
            intent = "retrieve"

        logger.info(f"[router] LLM classified → '{intent}' | reason: {reason}")
        return intent, reason

    except json.JSONDecodeError as e:
        logger.error(
            f"[router] JSON parse failed: {e} | raw output: '{raw}' | "
            f"falling back to keywords"
        )
        return _keyword_fallback(message), "json_parse_error"

    except Exception as e:
        logger.error(
            f"[router] LLM call failed: {e} | falling back to keywords"
        )
        return _keyword_fallback(message), "llm_error"


# ----------------------------------------------------------------
# LANGGRAPH NODE
# ----------------------------------------------------------------

async def clinical_router_node(state: AgentState) -> AgentState:
    """
    LangGraph node — classifies intent using the LLM
    and stores the result in state for the conditional edge.

    Reads from state:  message
    Writes to state:   intent
    """
    message = state.get("message", "")
    intent, reason = await classify_intent(message)

    logger.info(
        f"[clinical_router] '{message[:60]}' → intent='{intent}'"
    )

    return {
        **state,
        "intent": intent,
    }


def route_after_router(state: dict) -> str:
    """
    Conditional edge function — called by LangGraph after
    clinical_router_node to pick the next node.

    Returns the name of the next node as a string.
    """
    intent = state.get("intent", "retrieve")
    logger.info(f"[route_after_router] → {intent}")
    return intent