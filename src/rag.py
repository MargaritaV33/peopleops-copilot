from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.hybrid_retriever import hybrid_search


load_dotenv()

client = OpenAI()


MODEL_NAME = "gpt-5.4-mini"


SYSTEM_PROMPT = """
You are PeopleOps Copilot, an internal HR policy assistant.

Your job is to answer employee questions using ONLY the policy context
provided to you.

Rules:

1. Never invent company policies, eligibility rules, durations,
   reimbursement limits, procedures, or exceptions.

2. If the provided policy context is not sufficient to answer the question,
   clearly say that you could not find enough information in the available
   policy knowledge base.

3. When helpful, mention the relevant policy name or section naturally.

4. Do not claim that an HR action has been completed unless a tool has
   actually completed that action.

5. Keep answers clear, practical, and easy for an employee to understand.

6. Use bullet points or short steps when explaining a process.

7. Do not mention vector databases, chunks, embeddings, BM25, RRF,
   retrieval scores, or internal implementation details.

8. If the conversation history contains a follow-up question, use the
   previous turns to understand what the user is referring to.

9. Treat the provided policy context as the authoritative source for
   company-policy questions.
""".strip()



def format_context(results: list[dict]) -> str:
    """Convert retrieved policy results into structured LLM context."""

    context_blocks = []

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        policy_title = metadata.get(
            "policy_title",
            "Unknown Policy",
        )

        section = metadata.get(
            "section",
            "Not specified",
        )

        subsection = metadata.get(
            "subsection",
            "Not specified",
        )

        block = f"""
POLICY SOURCE {index}

Policy: {policy_title}
Section: {section}
Subsection: {subsection}

Policy text:
{result["document"]}
""".strip()

        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def extract_sources(
    results: list[dict],
) -> list[dict]:
    """Create a clean, deduplicated source list for the UI."""

    sources = []
    seen = set()

    for result in results:
        metadata = result["metadata"]

        source_key = (
            metadata.get("policy_title"),
            metadata.get("section"),
            metadata.get("subsection"),
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        sources.append(
            {
                "policy_title": metadata.get(
                    "policy_title",
                    "Unknown Policy",
                ),
                "section": metadata.get(
                    "section",
                ),
                "subsection": metadata.get(
                    "subsection",
                ),
                "source_url": metadata.get(
                    "source_url",
                ),
            }
        )

    return sources

def answer_question(
    query: str,
    conversation_history: Optional[list[dict]] = None,
    n_results: int = 5,
) -> dict:
    """
    Retrieve policy context and generate a grounded answer.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "retrieval_results": list[dict],
        }
    """

    retrieval_results = hybrid_search(
        query=query,
        n_results=n_results,
        candidate_k=10,
    )

    context = format_context(
        retrieval_results
    )

    sources = extract_sources(
        retrieval_results
    )

    messages = []

    if conversation_history:
        messages.extend(
            conversation_history[-10:]
        )

    current_prompt = f"""
EMPLOYEE QUESTION:

{query}


RETRIEVED POLICY CONTEXT:

{context}


Answer the employee's question using only the policy context above.
""".strip()

    messages.append(
        {
            "role": "user",
            "content": current_prompt,
        }
    )

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=messages,
        max_output_tokens=700,
    )

    return {
        "answer": response.output_text,
        "sources": sources,
        "retrieval_results": retrieval_results,
    }
