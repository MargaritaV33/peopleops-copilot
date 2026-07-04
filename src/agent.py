import json
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.tools import (
    create_hr_request_tool,
    get_policy_document,
    list_policy_topics,
    search_policy,
)


load_dotenv()

client = OpenAI()

MODEL_NAME = "gpt-5.4-mini"

AGENT_INSTRUCTIONS = """
You are PeopleOps Copilot, an employee policy and HR support assistant.

You have access to tools.

Tool-use rules:

1. For any question about company policy, benefits, leave, PTO,
   expenses, onboarding, development, remote work, or other employee
   handbook topics, use the search_policy tool before answering.

2. If the user asks which policies or topics are available,
   use list_policy_topics.

3. If the user asks for the original policy document, policy page,
   source, or link, use get_policy_document.

4. If the user explicitly asks to create, submit, log, or request
   an HR conversation or HR discussion request, use create_hr_request.

5. Never claim that a real HR appointment has been scheduled.
   The HR request workflow is a local demo action.

6. Do not invent company-policy information.

7. When search_policy returns policy context, answer only from that
   context.

8. If the available policy evidence is insufficient, say so clearly.

9. Keep responses practical, clear, and employee-friendly.

10. Do not mention BM25, vector search, embeddings, RRF, chunks,
    similarity scores, or other internal retrieval implementation details.

11. When the create_hr_request tool succeeds, briefly confirm that the
    demo request was created. Do not repeat the reference code, status,
    or priority in the prose response because the application interface
    displays those details separately.

12. Use recent conversation history to understand follow-up questions.

13. When a policy gives an approval threshold, exception threshold,
    or recommended amount, do not present it as a hard maximum or
    entitlement. Explain the distinction precisely.

14. If a user's question is ambiguous, distinguish between annual
    entitlement, recommended time off, consecutive leave, and approval
    requirements rather than collapsing them into one number.
""".strip()


AGENT_TOOLS = [
    {
        "type": "function",
        "name": "search_policy",
        "description": (
            "Search the employee policy knowledge base for evidence "
            "needed to answer a company-policy question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The employee's policy question or search query."
                    ),
                },
            },
            "required": [
                "query",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "create_hr_request",
        "description": (
            "Create a local demo HR discussion request when the employee "
            "explicitly asks to request or log a conversation with HR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "Short neutral topic for the HR discussion."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "A concise neutral summary of what the employee "
                        "would like to discuss."
                    ),
                },
                "priority": {
                    "type": "string",
                    "enum": [
                        "low",
                        "medium",
                        "high",
                    ],
                    "description": (
                        "Requested priority level."
                    ),
                },
            },
            "required": [
                "topic",
                "summary",
                "priority",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_policy_topics",
        "description": (
            "List the policy topics and documents currently available "
            "in the PeopleOps knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_policy_document",
        "description": (
            "Find the most relevant original policy pages and source "
            "links for a requested policy topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "The policy topic whose original document "
                        "or source page the employee wants."
                    ),
                },
            },
            "required": [
                "topic",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

def execute_tool(
    tool_name: str,
    arguments: dict,
) -> dict:
    """Route a model tool call to the correct Python function."""

    if tool_name == "search_policy":
        return search_policy(
            query=arguments["query"]
        )

    if tool_name == "create_hr_request":
        return create_hr_request_tool(
            topic=arguments["topic"],
            summary=arguments["summary"],
            priority=arguments["priority"],
        )

    if tool_name == "list_policy_topics":
        return list_policy_topics()

    if tool_name == "get_policy_document":
        return get_policy_document(
            topic=arguments["topic"]
        )

    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


def run_agent(
    query: str,
    conversation_history: Optional[list[dict]] = None,
    max_tool_rounds: int = 3,
) -> dict:
    """
    Run the PeopleOps agent with tool calling.

    Returns:
        {
            "answer": str,
            "sources": list,
            "actions": list,
            "tool_trace": list,
        }
    """

    input_list = []

    if conversation_history:
        input_list.extend(
            conversation_history[-10:]
        )

    input_list.append(
        {
            "role": "user",
            "content": query,
        }
    )

    collected_sources = []
    collected_actions = []
    tool_trace = []

    for _ in range(max_tool_rounds):
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=AGENT_INSTRUCTIONS,
            tools=AGENT_TOOLS,
            input=input_list,
            parallel_tool_calls=False,
            max_output_tokens=700,
        )

        # Preserve all model output items for the next request.
        # This is especially important for reasoning/tool-call items.
        input_list += response.output

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            return {
                "answer": response.output_text,
                "sources": collected_sources,
                "actions": collected_actions,
                "tool_trace": tool_trace,
            }

        for tool_call in tool_calls:
            arguments = json.loads(
                tool_call.arguments
            )

            tool_name = tool_call.name

            tool_trace.append(tool_name)

            tool_result = execute_tool(
                tool_name=tool_name,
                arguments=arguments,
            )

            if tool_name == "search_policy":
                collected_sources = tool_result.get(
                    "sources",
                    [],
                )

            if tool_name == "get_policy_document":
                collected_sources = tool_result.get(
                    "documents",
                    [],
                )

            if tool_name == "create_hr_request":
                collected_actions.append(
                    tool_result
                )

            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                }
            )

    return {
        "answer": (
            "I could not complete the request within "
            "the allowed tool-call steps."
        ),
        "sources": collected_sources,
        "actions": collected_actions,
        "tool_trace": tool_trace,
    }

# if __name__ == "__main__":
#     result = run_agent(
#         query=(
#             "What policy areas can you help me with?"
#         )
#     )

#     print("\n--- AGENT ANSWER ---\n")
#     print(result["answer"])

#     print("\n--- TOOL TRACE ---")
#     print(result["tool_trace"])

#     print("\n--- SOURCES ---")
#     print(result["sources"])