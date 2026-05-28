import pytest
import logging
from app.agents.graph_builder import run_rag_graph, build_rag_graph
from app.agents.clinical_agent import classify_intent, _keyword_fallback
from app.agents.safety_wrapper import apply_safety_wrapper
from app.rag.prompts.clinical_prompts import build_rag_messages
from app.tools.patient_timeline_tool import extract_dates_from_text
from app.tools.medication_checker_tool import quick_medication_scan


# ================================================================
# INTENT DETECTION — LLM based
# These make real Ollama calls — make sure ollama serve is running
# ================================================================

@pytest.mark.asyncio
async def test_intent_classifies_soap():
    """Standard SOAP request"""
    intent, reason = await classify_intent(
        "Write up a clinical note for this patient"
    )
    assert intent == "soap", f"Expected soap, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_soap_summarize():
    """Summarize phrasing should map to soap"""
    intent, reason = await classify_intent("Summarise the patient")
    assert intent == "soap", f"Expected soap, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_soap_document():
    """Documentation phrasing"""
    intent, reason = await classify_intent(
        "Draft the clinical documentation for this case"
    )
    assert intent == "soap", f"Expected soap, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_medication_formal():
    """Formal medication phrasing"""
    intent, reason = await classify_intent(
        "What medications is the patient currently taking?"
    )
    assert intent == "medication", f"Expected medication, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_medication_informal():
    """Informal phrasing — would break keyword routing"""
    intent, reason = await classify_intent("What is she on?")
    assert intent == "medication", f"Expected medication, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_medication_pull_up():
    """Colloquial clinical phrasing"""
    intent, reason = await classify_intent("Can you pull up her meds?")
    assert intent == "medication", f"Expected medication, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_medication_prescribed():
    """Prescription phrasing"""
    intent, reason = await classify_intent("What has been prescribed for him?")
    assert intent == "medication", f"Expected medication, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_timeline_formal():
    """Formal timeline request"""
    intent, reason = await classify_intent("Show me the patient timeline")
    assert intent == "timeline", f"Expected timeline, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_timeline_walk_through():
    """Narrative phrasing — would break keyword routing"""
    intent, reason = await classify_intent("Walk me through what happened")
    assert intent == "timeline", f"Expected timeline, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_timeline_history():
    """History phrasing"""
    intent, reason = await classify_intent("Give me the history on this one")
    assert intent == "timeline", f"Expected timeline, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_timeline_progression():
    """Progression phrasing"""
    intent, reason = await classify_intent(
        "How has the condition progressed over time?"
    )
    assert intent == "timeline", f"Expected timeline, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_retrieve_bp():
    """Specific factual question — blood pressure"""
    intent, reason = await classify_intent(
        "What was the blood pressure reading?"
    )
    assert intent == "retrieve", f"Expected retrieve, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_retrieve_labs():
    """Specific factual question — labs"""
    intent, reason = await classify_intent(
        "What did the lab results show?"
    )
    assert intent == "retrieve", f"Expected retrieve, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_retrieve_allergy():
    """Specific factual question — allergies"""
    intent, reason = await classify_intent(
        "Does the patient have any documented allergies?"
    )
    assert intent == "retrieve", f"Expected retrieve, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_classifies_retrieve_ecg():
    """Specific factual question — investigation"""
    intent, reason = await classify_intent("What did the ECG show?")
    assert intent == "retrieve", f"Expected retrieve, got {intent} | reason: {reason}"


@pytest.mark.asyncio
async def test_intent_returns_valid_intent_always():
    """
    Whatever the input, intent must always be one of the four
    valid values — never unknown or empty.
    """
    valid = {"retrieve", "medication", "timeline", "soap"}
    test_messages = [
        "hello",
        "???",
        "asdfghjkl",
        "tell me everything",
        "what do you think?",
    ]
    for msg in test_messages:
        intent, reason = await classify_intent(msg)
        assert intent in valid, (
            f"Got invalid intent '{intent}' for message: '{msg}'"
        )


@pytest.mark.asyncio
async def test_intent_reason_is_non_empty():
    """LLM should always provide a reason string"""
    intent, reason = await classify_intent("What medications is she taking?")
    assert isinstance(reason, str)
    assert len(reason) > 0


# ================================================================
# KEYWORD FALLBACK
# These are unit tests — no Ollama needed
# ================================================================

def test_keyword_fallback_medication():
    assert _keyword_fallback("what meds is she on") == "medication"


def test_keyword_fallback_timeline():
    assert _keyword_fallback("show me the timeline") == "timeline"


def test_keyword_fallback_soap():
    assert _keyword_fallback("give me a summary") == "soap"


def test_keyword_fallback_default_retrieve():
    """Anything unrecognised falls back to retrieve"""
    assert _keyword_fallback("something completely unrelated xyz") == "retrieve"


def test_keyword_fallback_empty_string():
    assert _keyword_fallback("") == "retrieve"


# ================================================================
# TIMELINE TOOL
# Unit tests — no Ollama needed
# ================================================================

def test_extract_dates_iso_format():
    text = "Patient admitted on 2024-01-15"
    dates = extract_dates_from_text(text)
    assert "2024-01-15" in dates


def test_extract_dates_slash_format():
    text = "Follow-up scheduled for 02/20/2024"
    dates = extract_dates_from_text(text)
    assert len(dates) >= 1


def test_extract_dates_multiple():
    text = "Admitted 2024-01-15. Discharged 2024-01-22."
    dates = extract_dates_from_text(text)
    assert len(dates) == 2


def test_extract_dates_none_found():
    dates = extract_dates_from_text("Patient has hypertension and diabetes")
    assert dates == []


def test_extract_dates_no_duplicates():
    text = "2024-01-15 admission. Readmitted 2024-01-15."
    dates = extract_dates_from_text(text)
    assert dates.count("2024-01-15") == 1


# ================================================================
# MEDICATION TOOL
# Unit tests — no Ollama needed
# ================================================================

def test_medication_scan_finds_known_drug():
    result = quick_medication_scan("Patient takes Metformin 500mg twice daily")
    assert result["has_medications"] is True
    assert "metformin" in result["found_keywords"]


def test_medication_scan_finds_dosage():
    result = quick_medication_scan("Amlodipine 5mg once daily")
    assert len(result["found_dosages"]) > 0
    assert "5mg" in result["found_dosages"][0] or "5" in result["found_dosages"][0]


def test_medication_scan_multiple_drugs():
    result = quick_medication_scan(
        "Patient takes Metformin 500mg and Amlodipine 5mg"
    )
    assert "metformin" in result["found_keywords"]
    assert "amlodipine" in result["found_keywords"]


def test_medication_scan_no_medications():
    result = quick_medication_scan("Patient reports fatigue and dizziness")
    assert result["has_medications"] is False
    assert result["found_keywords"] == []
    assert result["found_dosages"] == []


def test_medication_scan_case_insensitive():
    result = quick_medication_scan("METFORMIN prescribed at 500MG")
    assert result["has_medications"] is True


# ================================================================
# SAFETY WRAPPER
# Unit tests — no Ollama needed
# ================================================================

def test_safety_wrapper_returns_reply_unchanged():
    """Reply must be returned exactly — no appended text"""
    reply = "Patient takes Metformin 500mg daily"
    context = "Metformin 500mg twice daily prescribed"
    result = apply_safety_wrapper(reply, context)
    assert result == reply


def test_safety_wrapper_unverified_dosage_logged(caplog):
    """
    Dosage in reply not present in context should be
    logged as warning but reply must remain unchanged
    """
    reply = "Patient takes Aspirin 325mg daily"
    context = "Patient takes aspirin for pain relief"  # no dosage in context

    with caplog.at_level(logging.WARNING, logger="app.agents.safety_wrapper"):
        result = apply_safety_wrapper(reply, context)

    assert result == reply                    # reply unchanged
    assert "325mg" in caplog.text             # warning logged


def test_safety_wrapper_verified_dosage_no_warning(caplog):
    """Dosage present in context — no warning should be logged"""
    reply = "Patient takes Metformin 500mg daily"
    context = "Metformin 500mg twice daily as documented"

    with caplog.at_level(logging.WARNING, logger="app.agents.safety_wrapper"):
        result = apply_safety_wrapper(reply, context)

    assert result == reply
    assert "500mg" not in caplog.text         # no warning for verified dosage


def test_safety_wrapper_empty_reply():
    """Empty reply should come back as empty"""
    assert apply_safety_wrapper("", "some context") == ""


def test_safety_wrapper_empty_context():
    """Empty context — reply still comes back unchanged"""
    reply = "Patient has hypertension"
    assert apply_safety_wrapper(reply, "") == reply


# ================================================================
# GRAPH STRUCTURE
# Unit tests — no Ollama needed
# ================================================================

def test_graph_compiles():
    graph = build_rag_graph()
    assert graph is not None


def test_rag_messages_structure():
    messages = build_rag_messages(
        context="Patient has hypertension",
        question="What is the diagnosis?",
    )
    assert len(messages) == 2