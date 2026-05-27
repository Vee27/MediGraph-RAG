"""
Quick end-to-end test of the Day 5 ingest pipeline.
Run with: python notebooks/test_ingest.py

This is NOT a pytest test — it's a manual scratch script
so you can see exactly what each stage produces.
"""

import asyncio
import sys
import os

# Make sure Python can find the app package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.ingest.pdf_loader import load_pdf
from app.rag.ingest.chunking import chunk_pages
from app.rag.ingest.embedding import embed_chunks


async def main():
    PDF_PATH = "datasets/sample_charts/sample_chart.pdf"
    PATIENT_ID = "patient-001"

    print("=" * 60)
    print("STEP 1 — Load PDF")
    print("=" * 60)
    pages = load_pdf(PDF_PATH)
    print(f"Pages extracted: {len(pages)}")
    print(f"First 200 chars of page 1:\n{pages[0]['text'][:200]}")

    print()
    print("=" * 60)
    print("STEP 2 — Chunk pages")
    print("=" * 60)
    chunks = chunk_pages(pages, patient_id=PATIENT_ID)
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nSample chunk 0:")
    print(f"  text       : {chunks[0]['text'][:150]}")
    print(f"  patient_id : {chunks[0]['patient_id']}")
    print(f"  page       : {chunks[0]['page']}")
    print(f"  chunk_index: {chunks[0]['chunk_index']}")

    print()
    print("=" * 60)
    print("STEP 3 — Embed first 2 chunks only (saves time)")
    print("=" * 60)
    sample_chunks = chunks[:2]
    embedded = await embed_chunks(sample_chunks)

    for ec in embedded:
        vector = ec["embedding"]
        print(f"\nChunk {ec['chunk_index']}:")
        print(f"  text preview  : {ec['text'][:80]}...")
        print(f"  vector length : {len(vector)}")
        print(f"  first 5 values: {[round(v, 4) for v in vector[:5]]}")

    print()
    print("=" * 60)
    print("ALL STEPS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())