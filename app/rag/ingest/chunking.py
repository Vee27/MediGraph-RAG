import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# How we split text — these numbers are tuned for medical documents
CHUNK_SIZE = 512      # Each chunk is at most 512 characters
CHUNK_OVERLAP = 64    # Neighbouring chunks share 64 characters
                      # This prevents important sentences being cut in half


def chunk_pages(pages: list[dict], patient_id: str) -> list[dict]:
    """
    Take the pages extracted by pdf_loader and split them into chunks.

    Args:
        pages:      Output from load_pdf() — list of {page, text}
        patient_id: Unique ID for this patient — stored with every chunk
                    so we can filter by patient later

    Returns:
        List of chunk dicts:
        [
            {
                "text": "Patient has hypertension...",
                "patient_id": "patient-001",
                "page": 1,
                "chunk_index": 0,
            },
            ...
        ]

    Why RecursiveCharacterTextSplitter?
        It tries to split on natural boundaries first:
        paragraph → sentence → word → character
        So chunks are more readable than hard character cuts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_index = 0

    for page in pages:
        page_num = page["page"]
        text = page["text"]

        # Split this page's text into chunks
        raw_chunks = splitter.split_text(text)

        logger.info(f"Page {page_num}: split into {len(raw_chunks)} chunk(s)")

        for raw_chunk in raw_chunks:
            # Skip chunks that are too short to be useful
            if len(raw_chunk.strip()) < 30:
                continue

            all_chunks.append({
                "text": raw_chunk.strip(),
                "patient_id": patient_id,
                "page": page_num,
                "chunk_index": chunk_index,
            })

            chunk_index += 1

    logger.info(f"Total chunks created: {len(all_chunks)} for patient: {patient_id}")
    return all_chunks