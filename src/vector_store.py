from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.chunking import create_all_chunks

from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"

COLLECTION_NAME = "peopleops_policies"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load and cache the local SentenceTransformer model."""

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    return model



def get_chroma_client():
    """Create a persistent local ChromaDB client."""

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
    )

    return client


def build_vector_store():
    """Create embeddings for all policy chunks and store them in ChromaDB."""

    chunks = create_all_chunks()

    print(f"\nPreparing {len(chunks)} chunks for embedding...")

    model = load_embedding_model()

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    ids = [
        f"chunk_{index:04d}"
        for index in range(len(chunks))
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = get_chroma_client()

    # Rebuild cleanly during development.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print("\nVector store created successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")

    return collection

def semantic_search(
    query: str,
    n_results: int = 5,
):
    """Search ChromaDB using semantic similarity."""

    model = load_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    client = get_chroma_client()

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=n_results,
    )

    return results
