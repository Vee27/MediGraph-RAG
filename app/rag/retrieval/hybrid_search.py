import logging
from rank_bm25 import BM25Okapi
from app.rag.ingest.vector_store import vector_store

logger = logging.getLogger(__name__)


def _bm25_search(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    BM25 keyword search over a list of chunks.

    BM25 is a classic information retrieval algorithm.
    It finds chunks that contain words from the query,
    scoring them higher if the words appear more often
    and the chunk is shorter (more focused).

    Args:
        query:  The search question
        chunks: All chunks for this patient (from ChromaDB)
        top_k:  How many top results to return

    Returns:
        Top-k chunks sorted by BM25 score, highest first
    """
    if not chunks:
        return []

    # Tokenise — split each chunk into individual words
    tokenised_corpus = [c["text"].lower().split() for c in chunks]
    tokenised_query = query.lower().split()

    # Build BM25 index from the corpus
    bm25 = BM25Okapi(tokenised_corpus)

    # Score each chunk against the query
    scores = bm25.get_scores(tokenised_query)

    # Pair each chunk with its score and sort
    scored = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Return top_k with bm25_score added
    results = []
    for chunk, score in scored[:top_k]:
        results.append({
            **chunk,
            "bm25_score": round(float(score), 4),
        })

    logger.info(f"BM25 returned {len(results)} results")
    return results


def _combine_results(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Merge dense (semantic) and BM25 (keyword) results.

    Strategy — Reciprocal Rank Fusion (RRF):
    Each chunk gets a score based on its rank in each list.
    A chunk ranked #1 in both lists scores highest overall.
    This is simple, effective, and doesn't need weight tuning.

    RRF formula: score = 1/(rank + 60) for each list
    The 60 is a constant that prevents very high scores
    for top-ranked items dominating everything else.

    Args:
        dense_results: From ChromaDB vector search
        bm25_results:  From BM25 keyword search
        top_k:         Final number of results to return

    Returns:
        Merged and re-ranked list of chunks
    """
    # Use chunk_index as the unique key to identify chunks
    scores = {}
    chunk_map = {}

    # Score dense results by rank
    for rank, chunk in enumerate(dense_results):
        key = chunk.get("chunk_index", rank)
        rrf_score = 1 / (rank + 60)
        scores[key] = scores.get(key, 0) + rrf_score
        chunk_map[key] = chunk

    # Score BM25 results by rank
    for rank, chunk in enumerate(bm25_results):
        key = chunk.get("chunk_index", rank)
        rrf_score = 1 / (rank + 60)
        scores[key] = scores.get(key, 0) + rrf_score
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Sort by combined RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = chunk_map[key]
        chunk["hybrid_score"] = round(scores[key], 6)
        results.append(chunk)

    logger.info(f"Hybrid search combined into {len(results)} results")
    return results


async def hybrid_search(
    query: str,
    patient_id: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Main entry point — runs dense + BM25 search and combines them.

    Args:
        query:      The user's question
        patient_id: Which patient's chunks to search
        top_k:      How many results to return

    Returns:
        Top-k most relevant chunks, sorted by hybrid score
    """
    logger.info(f"Hybrid search — query: '{query[:60]}', patient: {patient_id}")

    # 1. Dense semantic search via ChromaDB
    dense_results = await vector_store.query_chunks(
        query_text=query,
        patient_id=patient_id,
        top_k=top_k * 2,  # fetch more so BM25 has something to work with
    )

    if not dense_results:
        logger.warning(f"No chunks found for patient: {patient_id}")
        return []

    # 2. BM25 keyword search over the same chunks
    bm25_results = _bm25_search(
        query=query,
        chunks=dense_results,
        top_k=top_k,
    )

    # 3. Combine both using RRF
    combined = _combine_results(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=top_k,
    )

    return combined