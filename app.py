import os

from dotenv import load_dotenv

from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent import run_agent
from src.sources import POLICY_SOURCES
from src.vector_store import (
    COLLECTION_NAME,
    get_chroma_client,
)

load_dotenv()

CAL_BOOKING_URL = os.getenv(
    "CAL_BOOKING_URL",
    "",
)

st.set_page_config(
    page_title="PeopleOps Copilot",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)



# custom visual styling:

st.markdown(
    """
    <style>
    /* Main application */
    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(91, 124, 250, 0.10),
                transparent 28%
            ),
            linear-gradient(
                180deg,
                #0b0e13 0%,
                #11151c 100%
            );
    }

    /* Main content width */
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hero label */
    .eyebrow {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8aa4ff;
        margin-bottom: 0.6rem;
    }

    /* Hero title */
    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4.3rem);
        font-weight: 800;
        line-height: 0.96;
        letter-spacing: -0.045em;
        margin-bottom: 1rem;
        color: #f5f7fb;
    }

    .hero-subtitle {
        max-width: 780px;
        font-size: 1.08rem;
        line-height: 1.65;
        color: #adb7c6;
        margin-bottom: 1.5rem;
    }

    /* Status */
    .status-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 0.8rem 0 1.8rem 0;
    }

    .status-pill {
        padding: 0.42rem 0.75rem;
        border: 1px solid #303847;
        border-radius: 999px;
        background: rgba(23, 29, 39, 0.78);
        color: #c7cfdb;
        font-size: 0.82rem;
    }

    .status-live {
        color: #8fe0a1;
    }

    /* Section label */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8490a1;
        margin-top: 1.8rem;
        margin-bottom: 0.7rem;
    }

    /* Metric boxes */
    div[data-testid="stMetric"] {
        background: rgba(20, 25, 34, 0.92);
        border: 1px solid #2c3441;
        padding: 1rem 1.1rem;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8f9baa;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        border: 1px solid #303949;
        background: #171d27;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        border-color: #718cff;
        transform: translateY(-1px);
    }

    /* Chat */
    div[data-testid="stChatMessage"] {
        border: 1px solid #29313e;
        border-radius: 16px;
        padding: 0.35rem 0.6rem;
        background: rgba(19, 24, 32, 0.82);
        margin-bottom: 0.8rem;
    }

    /* Source label */
    .source-title {
        color: #dfe5ef;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .source-path {
        color: #8e99aa;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    /* Action success */
    .action-card {
        border: 1px solid rgba(77, 193, 115, 0.42);
        background: rgba(43, 120, 67, 0.12);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 0.7rem;
    }

    .action-reference {
        font-size: 1.35rem;
        font-weight: 800;
        color: #90e0a5;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #252c37;
    }

    /* Hide Streamlit branding/footer */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# knowledge base statistics:

@st.cache_data
def get_knowledge_base_stats():
    """Read knowledge-base statistics from the persisted Chroma collection."""

    client = get_chroma_client()

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    collection_data = collection.get(
        include=["metadatas"]
    )

    metadatas = collection_data.get(
        "metadatas",
        [],
    )

    category_counts = Counter(
        metadata.get(
            "category",
            "other",
        )
        for metadata in metadatas
    )

    return {
        "policy_count": len(POLICY_SOURCES),
        "chunk_count": collection.count(),
        "category_count": len(category_counts),
        "category_counts": dict(category_counts),
    }


stats = get_knowledge_base_stats()

# session state:


if "messages" not in st.session_state:
    st.session_state.messages = []


if "last_tool_trace" not in st.session_state:
    st.session_state.last_tool_trace = []




def render_sources(
    sources: list[dict],
):
    """Render retrieved source metadata beneath an answer."""

    if not sources:
        return

    with st.expander(
        f"Sources used · {len(sources)}",
        expanded=False,
    ):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            policy_title = source.get(
                "policy_title",
                "Policy",
            )

            section = source.get("section")
            subsection = source.get("subsection")
            source_url = source.get("source_url")

            path_parts = [
                part
                for part in [
                    section,
                    subsection,
                ]
                if part
            ]

            path = " → ".join(path_parts)

            st.markdown(
                f"**{index}. {policy_title}**"
            )

            if path:
                st.caption(path)

            if source_url:
                st.markdown(
                    f"[Open original policy ↗]({source_url})"
                )

            if index < len(sources):
                st.divider()

def render_actions(
    actions: list[dict],
):
    """Render successful application actions."""

    for action in actions:
        reference_code = action.get(
            "reference_code",
            "Unknown",
        )

        status = action.get(
            "status",
            "Pending",
        )

        priority = action.get(
            "priority",
            "medium",
        )

        st.html(
            f"""
            <div class="action-card">
                <div style="
                    font-size: 0.72rem;
                    letter-spacing: 0.11em;
                    text-transform: uppercase;
                    color: #7ccf91;
                    margin-bottom: 0.35rem;
                ">
                    Demo HR request created
                </div>

                <div class="action-reference">
                    {reference_code}
                </div>

                <div style="
                    color: #aeb9c8;
                    margin-top: 0.35rem;
                ">
                    Status: {status}
                    &nbsp;·&nbsp;
                    Priority: {priority.title()}
                </div>
            </div>
            """
        )
        if CAL_BOOKING_URL:
            st.link_button(
                "📅 Book an HR conversation",
                CAL_BOOKING_URL,
                use_container_width=True,
            )


def render_assistant_payload(
    message: dict,
):
    """Render an assistant answer and its related UI elements."""

    st.markdown(
        message["content"]
    )

    render_actions(
        message.get(
            "actions",
            [],
        )
    )

    render_sources(
        message.get(
            "sources",
            [],
        )
    )

def build_agent_history() -> list[dict]:
    """Convert UI messages into clean role/content conversation history."""

    clean_history = []

    for message in st.session_state.messages:
        avatar = (
            ":material/person:"
            if message["role"] == "user"
            else ":material/support_agent:"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):
            if message["role"] == "assistant":
                render_assistant_payload(
                    message
                )
            else:
                st.markdown(
                    message["content"]
                )

    return clean_history[-10:]

with st.sidebar:
    st.markdown(
        "### PeopleOps Copilot"
    )

    st.caption(
        "Grounded policy intelligence + HR action workflows"
    )

    st.divider()

    st.markdown(
        "#### Knowledge coverage"
    )

    coverage_df = pd.DataFrame(
        [
            {
                "Category": category.replace(
                    "_",
                    " ",
                ).title(),
                "Chunks": count,
            }
            for category, count
            in stats[
                "category_counts"
            ].items()
        ]
    ).sort_values(
        "Chunks",
        ascending=True,
    )

    figure = px.bar(
        coverage_df,
        x="Chunks",
        y="Category",
        orientation="h",
        height=360,
    )

    figure.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    st.divider()

    st.markdown(
        "#### Retrieval architecture"
    )

    st.caption(
        "BM25 lexical search  +  semantic vector search  →  Reciprocal Rank Fusion  →  grounded generation"
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_tool_trace = []
        st.rerun()

st.html(
    """
    <div class="eyebrow">
        Employee Policy Intelligence
    </div>

    <div class="hero-title">
        PeopleOps<br>Copilot
    </div>

    <div class="hero-subtitle">
        Ask company-policy questions, find original documents,
        explore employee guidance, and create demo HR discussion
        requests — all through one grounded assistant.
    </div>

    <div class="status-row">
        <span class="status-pill status-live">
            ● Knowledge Base Ready
        </span>

        <span class="status-pill">
            Hybrid Retrieval
        </span>

        <span class="status-pill">
            Agent Tools
        </span>

        <span class="status-pill">
            Source Grounded
        </span>
    </div>
    """
)



metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Policies",
        stats["policy_count"],
    )

with metric_2:
    st.metric(
        "Indexed chunks",
        stats["chunk_count"],
    )

with metric_3:
    st.metric(
        "Categories",
        stats["category_count"],
    )

with metric_4:
    st.metric(
        "Retrieval",
        "Hybrid + RRF",
    )


st.markdown(
    '<div class="section-label">Quick actions</div>',
    unsafe_allow_html=True,
)


quick_prompts = {
    "🏖 Plan time off": (
        "I want to take three days of PTO. "
        "What should I do and how much notice should I give?"
    ),
    "👶 Parental leave": (
        "Explain the parental leave options "
        "and how the leave can be scheduled."
    ),
    "✈ Expenses": (
        "Can I get reimbursed for using a coworking space?"
    ),
    "👋 New employee": (
        "What should a new employee focus on during onboarding?"
    ),
    "◌ Remote work": (
        "How should remote teams communicate asynchronously?"
    ),
    "💬 Talk to HR": (
        "I would like to request a conversation "
        "with HR about workload and wellbeing."
    ),
}


prompt_to_run = None

button_columns = st.columns(3)

for index, (
    label,
    prompt_text,
) in enumerate(quick_prompts.items()):

    column = button_columns[index % 3]

    with column:
        if st.button(
            label,
            key=f"quick_prompt_{index}",
            use_container_width=True,
        ):
            prompt_to_run = prompt_text


st.markdown(
    '<div class="section-label">Conversation</div>',
    unsafe_allow_html=True,
)


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        if message["role"] == "assistant":
            render_assistant_payload(
                message
            )
        else:
            st.markdown(
                message["content"]
            )


typed_prompt = st.chat_input(
    "Ask about PTO, leave, benefits, expenses, onboarding..."
)


user_prompt = (
    typed_prompt
    if typed_prompt
    else prompt_to_run
)


if user_prompt:
    user_message = {
        "role": "user",
        "content": user_prompt,
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    conversation_history = build_agent_history()

    with st.chat_message(
        "assistant",
        avatar=":material/support_agent:",
    ):
        with st.spinner(
            "Searching policies and reasoning..."
        ):
            try:
                result = run_agent(
                    query=user_prompt,
                    conversation_history=conversation_history,
                )

                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get(
                        "sources",
                        [],
                    ),
                    "actions": result.get(
                        "actions",
                        [],
                    ),
                    "tool_trace": result.get(
                        "tool_trace",
                        [],
                    ),
                }

                st.session_state.messages.append(
                    assistant_message
                )

                st.session_state.last_tool_trace = (
                    assistant_message[
                        "tool_trace"
                    ]
                )

                render_assistant_payload(
                    assistant_message
                )

            except Exception as error:
                st.error(
                    "The assistant could not complete "
                    "the request."
                )

                with st.expander(
                    "Technical details"
                ):
                    st.code(str(error))