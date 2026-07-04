import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.chunking import create_all_chunks
from src.vector_store import semantic_search


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase and tokenize text for BM25 lexical retrieval."""

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


@lru_cache(maxsize=1)
def build_bm25_index():
    """Build and cache the BM25 index over all policy chunks."""

    chunks = create_all_chunks()

    tokenized_corpus = [
        tokenize_for_bm25(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return chunks, bm25


def bm25_search(
    query: str,
    n_results: int = 10,
) -> list[dict]:
    """Return top BM25 lexical-search results."""

    chunks, bm25 = build_bm25_index()

    query_tokens = tokenize_for_bm25(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:n_results]

    results = []

    for index in ranked_indices:
        results.append(
            {
                "id": f"chunk_{index:04d}",
                "document": chunks[index].page_content,
                "metadata": chunks[index].metadata,
                "bm25_score": float(scores[index]),
            }
        )

    return results


def dense_search(
    query: str,
    n_results: int = 10,
) -> list[dict]:
    """Convert Chroma semantic-search output into a simple ranked list."""

    raw_results = semantic_search(
        query=query,
        n_results=n_results,
    )

    results = []

    for rank in range(
        len(raw_results["documents"][0])
    ):
        results.append(
            {
                "id": raw_results["ids"][0][rank],
                "document": raw_results["documents"][0][rank],
                "metadata": raw_results["metadatas"][0][rank],
                "distance": raw_results["distances"][0][rank],
            }
        )

    return results


def reciprocal_rank_fusion(
    result_sets: dict[str, list[dict]],
    n_results: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion."""

    fused = {}

    for method, results in result_sets.items():
        for rank, item in enumerate(
            results,
            start=1,
        ):
            chunk_id = item["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "id": chunk_id,
                    "document": item["document"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "retrieval_methods": [],
                }

            fused[chunk_id]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

            fused[chunk_id][
                "retrieval_methods"
            ].append(method)

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:n_results]


def hybrid_search(
    query: str,
    n_results: int = 5,
    candidate_k: int = 10,
) -> list[dict]:
    """Run BM25 and dense search, then fuse results with RRF."""

    bm25_results = bm25_search(
        query=query,
        n_results=candidate_k,
    )

    dense_results = dense_search(
        query=query,
        n_results=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        result_sets={
            "bm25": bm25_results,
            "dense": dense_results,
        },
        n_results=n_results,
    )

    return fused_results
