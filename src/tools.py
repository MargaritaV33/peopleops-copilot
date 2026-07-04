from src.database import create_hr_request
from src.hybrid_retriever import hybrid_search
from src.rag import extract_sources, format_context
from src.sources import POLICY_SOURCES


def search_policy(
    query: str,
) -> dict:
    """
    Retrieve policy context for an employee question.

    This tool performs retrieval only.
    The agent uses the retrieved evidence to write the final answer.
    """

    results = hybrid_search(
        query=query,
        n_results=5,
        candidate_k=10,
    )

    return {
        "context": format_context(results),
        "sources": extract_sources(results),
    }


def create_hr_request_tool(
    topic: str,
    summary: str,
    priority: str,
) -> dict:
    """Create a demo HR discussion request."""

    return create_hr_request(
        topic=topic,
        summary=summary,
        priority=priority,
    )


def list_policy_topics() -> dict:
    """Return all available policy topics in the knowledge base."""

    policies = [
        {
            "title": source["title"],
            "category": source["category"],
            "source_url": source["page_url"],
        }
        for source in POLICY_SOURCES
    ]

    return {
        "policy_count": len(policies),
        "policies": policies,
    }


def get_policy_document(
    topic: str,
) -> dict:
    """
    Find relevant original policy pages for a topic.
    """

    results = hybrid_search(
        query=topic,
        n_results=5,
        candidate_k=10,
    )

    documents = []
    seen_urls = set()

    for result in results:
        metadata = result["metadata"]

        source_url = metadata.get(
            "source_url"
        )

        if not source_url:
            continue

        if source_url in seen_urls:
            continue

        seen_urls.add(source_url)

        documents.append(
            {
                "policy_title": metadata.get(
                    "policy_title"
                ),
                "section": metadata.get(
                    "section"
                ),
                "subsection": metadata.get(
                    "subsection"
                ),
                "source_url": source_url,
            }
        )

    return {
        "topic": topic,
        "documents": documents[:3],
    }