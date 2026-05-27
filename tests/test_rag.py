import pytest
import asyncio
from app.rag.ingest.pdf_loader import load_pdf
from app.rag.ingest.chunking import chunk_pages
from app.rag.ingest.embedding import embed_chunks
from app.rag.ingest.vector_store import vector_store

PDF_PATH = "datasets/sample_charts/sample_chart.pdf"
TEST_PATIENT = "pytest-patient"


def test_pdf_loader():
    """PDF loader should extract at least one page"""
    pages = load_pdf(PDF_PATH)
    assert len(pages) > 0
    assert "text" in pages[0]
    assert len(pages[0]["text"]) > 10


def test_chunking():
    """Chunker should produce multiple chunks from a page"""
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages, patient_id=TEST_PATIENT)
    assert len(chunks) > 0
    assert chunks[0]["patient_id"] == TEST_PATIENT
    assert "text" in chunks[0]
    assert "page" in chunks[0]
    assert "chunk_index" in chunks[0]


@pytest.mark.asyncio
async def test_embedding():
    """Embedding should return a vector of length 768"""
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages, patient_id=TEST_PATIENT)
    embedded = await embed_chunks(chunks[:1])  # only embed first chunk
    assert len(embedded) == 1
    assert "embedding" in embedded[0]
    assert len(embedded[0]["embedding"]) == 768


@pytest.mark.asyncio
async def test_vector_store_upsert_and_query():
    """Store chunks then query — results should not be empty"""
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages, patient_id=TEST_PATIENT)
    embedded = await embed_chunks(chunks)

    count = vector_store.upsert_chunks(embedded)
    assert count > 0

    results = await vector_store.query_chunks(
        query_text="What is the patient diagnosed with?",
        patient_id=TEST_PATIENT,
        top_k=3,
    )
    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]
    assert results[0]["score"] > 0