import chromadb
import logging
from pathlib import Path
from app.config.settings import settings
from app.models.ollama_client import ollama_client

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around ChromaDB.

    Every patient gets their own collection inside ChromaDB.
    A collection is like a table in a normal database —
    it holds all the chunks and their vectors for one patient.

    Structure:
        ChromaDB
        ├── collection: patient-001
        │   ├── id: chunk_0, embedding: [...], document: "text...", metadata: {page: 1}
        │   ├── id: chunk_1, embedding: [...], document: "text...", metadata: {page: 1}
        └── collection: patient-002
            └── ...
    """

    def __init__(self):
        # Create the folder if it doesn't exist
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # PersistentClient saves data to disk automatically
        # Every time you restart the app, your data is still there
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        logger.info(f"ChromaDB initialised at: {persist_dir}")

    def _get_collection(self, patient_id: str):
        """
        Get or create a ChromaDB collection for a patient.
        If the collection already exists, it just returns it.
        If not, it creates a new one.

        Collection names can only have letters, numbers, and hyphens.
        """
        collection_name = f"patient-{patient_id}".replace("_", "-").lower()

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            # cosine similarity — best for text embeddings
        )

        logger.info(f"Using collection: {collection_name}")
        return collection

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """
        Store chunks and their embeddings in ChromaDB.

        'Upsert' means insert or update — if a chunk with the
        same ID already exists, it gets replaced. This means
        you can safely re-upload a chart without duplicates.

        Args:
            chunks: Output from embed_chunks() — each chunk must
                    have 'embedding', 'text', 'patient_id',
                    'page', 'chunk_index' fields

        Returns:
            Number of chunks stored
        """
        if not chunks:
            logger.warning("No chunks to store")
            return 0

        # All chunks must belong to the same patient
        patient_id = chunks[0]["patient_id"]
        collection = self._get_collection(patient_id)

        # ChromaDB expects separate lists for ids, embeddings, documents, metadata
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Unique ID for each chunk — patient + chunk index
            chunk_id = f"{patient_id}-chunk-{chunk['chunk_index']}"

            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadatas.append({
                "patient_id": patient_id,
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            })

        # Store everything in one go
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Stored {len(chunks)} chunks for patient: {patient_id}")
        return len(chunks)

    async def query_chunks(
        self,
        query_text: str,
        patient_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search ChromaDB for chunks most relevant to the query.

        Steps:
        1. Embed the query text into a vector
        2. Compare it to all stored vectors for this patient
        3. Return the top_k most similar chunks

        Args:
            query_text: The question or search text
            patient_id: Only search this patient's chunks
            top_k:      How many chunks to return (default 5)

        Returns:
            List of dicts:
            [
                {
                    "text": "Patient has hypertension...",
                    "page": 1,
                    "chunk_index": 0,
                    "score": 0.92,   ← similarity score, higher = more relevant
                },
                ...
            ]
        """
        collection = self._get_collection(patient_id)

        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            logger.warning(f"No chunks found for patient: {patient_id}")
            return []

        logger.info(f"Searching {count} chunks for patient: {patient_id}")

        # Embed the query — same model used for chunks
        query_vector = await ollama_client.embed(query_text)

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),  # can't return more than we have
            include=["documents", "metadatas", "distances"],
        )

        # Format the results into clean dicts
        chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # ChromaDB returns distance (lower = more similar)
            # Convert to score (higher = more similar) for readability
            score = round(1 - distance, 4)

            chunks.append({
                "text": doc,
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "patient_id": patient_id,
                "score": score,
            })

        logger.info(f"Found {len(chunks)} relevant chunks")
        return chunks

    def delete_patient(self, patient_id: str) -> bool:
        """
        Delete all chunks for a patient.
        Useful if a chart needs to be re-uploaded from scratch.
        """
        try:
            collection_name = f"patient-{patient_id}".replace("_", "-").lower()
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection for patient: {patient_id}")
            return True
        except Exception as e:
            logger.error(f"Could not delete patient {patient_id}: {e}")
            return False

    def get_patient_chunk_count(self, patient_id: str) -> int:
        """How many chunks are stored for a patient."""
        try:
            collection = self._get_collection(patient_id)
            return collection.count()
        except Exception:
            return 0


vector_store = VectorStore()