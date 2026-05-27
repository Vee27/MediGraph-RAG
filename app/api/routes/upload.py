import logging
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.response_models import UploadResponse
from app.rag.ingest.pdf_loader import load_pdf
from app.rag.ingest.chunking import chunk_pages
from app.rag.ingest.embedding import embed_chunks
from app.rag.ingest.vector_store import vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_chart(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
):
    """
    Upload a PDF patient chart and index it for RAG.

    This runs the full pipeline automatically:
    PDF → extract text → chunk → embed → store in ChromaDB

    Args:
        file:       The PDF file (sent as multipart form data)
        patient_id: Unique identifier for this patient
                    e.g. "patient-001" or "john-doe-dob-1990"

    Returns:
        How many chunks were stored and pages processed

    How to call this:
        curl -X POST http://localhost:8000/upload
          -F "file=@your_chart.pdf"
          -F "patient_id=patient-001"
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    logger.info(f"Upload received — file: {file.filename}, patient: {patient_id}")

    # We need to save the uploaded file temporarily to disk
    # because pdf_loader needs a file path, not raw bytes
    # tempfile creates a temp file and cleans it up automatically
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    try:
        # STEP 1 — Extract text from PDF
        logger.info("Step 1/4: Extracting text from PDF...")
        pages = load_pdf(tmp_path)

        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from this PDF. It may be scanned or image-based."
            )

        # STEP 2 — Split into chunks
        logger.info("Step 2/4: Splitting into chunks...")
        chunks = chunk_pages(pages, patient_id=patient_id)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not create chunks from this document."
            )

        # STEP 3 — Create embeddings
        logger.info("Step 3/4: Creating embeddings (this may take a minute)...")
        embedded_chunks = await embed_chunks(chunks)

        # STEP 4 — Store in ChromaDB
        logger.info("Step 4/4: Storing in ChromaDB...")
        chunks_stored = vector_store.upsert_chunks(embedded_chunks)

        logger.info(
            f"Upload complete — patient: {patient_id}, "
            f"pages: {len(pages)}, chunks: {chunks_stored}"
        )

        return UploadResponse(
            message=f"Chart uploaded and indexed successfully",
            patient_id=patient_id,
            chunks_stored=chunks_stored,
            pages_processed=len(pages),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

    finally:
        # Always clean up the temp file even if something went wrong
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Temp file cleaned up")