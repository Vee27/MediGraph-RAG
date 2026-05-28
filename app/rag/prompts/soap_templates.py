SOAP_SYSTEM_PROMPT = """You are a clinical documentation specialist.
Generate structured SOAP notes from patient chart data.

Rules:
- Use ONLY information present in the provided context
- If a section has no data, write: Not documented
- Use precise clinical language
- Include specific values where present (vitals, lab results, dosages)
- Be concise — each section should be 2-5 bullet points maximum
"""

SOAP_NOTE_TEMPLATE = """Using the patient chart context below, \
generate a structured SOAP note.

PATIENT CHART CONTEXT:
{context}

Generate the SOAP note in exactly this format:

SUBJECTIVE:
- (Patient-reported symptoms, complaints, history)

OBJECTIVE:
- (Vitals, lab values, physical exam findings)

ASSESSMENT:
- (Diagnosis or clinical impression)

PLAN:
- (Treatment, medications, follow-up)
"""

MEDICATION_SYSTEM_PROMPT = """You are a clinical pharmacist assistant.
Extract and list all medications from patient chart text.

Rules:
- Only list medications explicitly present in the context
- Include name, dosage, and frequency for each
- Format as a clean structured list
- If a field is absent, write: not specified
"""

MEDICATION_TEMPLATE = """From the patient chart context below,
identify and list all medications.

PATIENT CHART CONTEXT:
{context}

Format your response as:

MEDICATIONS:
1. [Name] | [Dosage] | [Frequency]
2. [Name] | [Dosage] | [Frequency]

If no medications are documented, state that clearly.
"""

TIMELINE_SYSTEM_PROMPT = """You are a clinical timeline specialist.
Extract and organise chronological events from patient chart text.

Rules:
- Only include events explicitly mentioned in the context
- Sort from oldest to most recent
- If no dates are found, organise by clinical sequence
- Be factual and concise
"""

TIMELINE_TEMPLATE = """From the patient chart context below,
extract all clinical events in chronological order.

PATIENT CHART CONTEXT:
{context}

Format your response as:

PATIENT TIMELINE:
[Date/Period] — [Clinical event]
[Date/Period] — [Clinical event]

If no specific dates are found, note that and list \
events in clinical order.
"""


def build_soap_messages(context: str) -> list[dict]:
    return [
        {"role": "system", "content": SOAP_SYSTEM_PROMPT},
        {"role": "user", "content": SOAP_NOTE_TEMPLATE.format(context=context)},
    ]


def build_medication_messages(context: str) -> list[dict]:
    return [
        {"role": "system", "content": MEDICATION_SYSTEM_PROMPT},
        {"role": "user", "content": MEDICATION_TEMPLATE.format(context=context)},
    ]


def build_timeline_messages(context: str) -> list[dict]:
    return [
        {"role": "system", "content": TIMELINE_SYSTEM_PROMPT},
        {"role": "user", "content": TIMELINE_TEMPLATE.format(context=context)},
    ]