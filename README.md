# PeopleOps Copilot

> A one-day AI Engineering project that became a complete employee-policy assistant: policy ingestion, cleaning, structure-aware chunking, local embeddings, ChromaDB, BM25, Reciprocal Rank Fusion, grounded RAG, LLM tool calling, SQLite persistence, and a Streamlit interface.

---

## What the project does

PeopleOps Copilot is an internal HR policy assistant built on a curated set of public GitLab Handbook pages.

Employees can:

- ask policy questions in natural language;
- receive answers grounded in retrieved policy text;
- inspect the original policy sources used;
- request the original policy page for a topic;
- list the policy areas available in the knowledge base;
- create a **local demo HR discussion request** stored in SQLite;
- optionally open a configurable booking page through `CAL_BOOKING_URL`.

The application is explicit about one important limitation: the local HR request workflow is a demonstration. It does not contact a real HR department or schedule a real appointment by itself.

---

## Why this project was built

The assignment was to design and implement an end-to-end chatbot application and make the project available in a public GitHub repository.

The project intentionally combines medium and difficult components from the course:

- document ingestion;
- document cleaning;
- text chunking;
- embeddings;
- vector storage;
- conversational memory;
- dense semantic retrieval;
- BM25 sparse retrieval;
- Reciprocal Rank Fusion;
- grounded LLM generation;
- agent tool calling;
- SQLite persistence;
- Streamlit UI;
- source transparency;
- policy-specific guardrails.

The goal was not to hide the full workflow inside a single framework call. The individual stages are explicit so the architecture can be understood, tested, debugged, and explained.

---

# Architecture

```text
PUBLIC GITLAB HANDBOOK MARKDOWN
              |
              v
        src/sources.py
         source registry
              |
              v
        src/ingest.py
          download files
              |
              v
        data/raw/*.md
              |
              v
       src/cleaning.py
  - front matter removal
  - Hugo shortcode cleanup
  - selected HTML wrapper cleanup
  - inline Markdown cleanup
  - whitespace normalization
              |
              v
       src/chunking.py
  MarkdownHeaderTextSplitter
              |
              v
 RecursiveCharacterTextSplitter
              |
              v
  source + section metadata
              |
              v
     low-value chunk filter
              |
      +-------+-------+
      |               |
      v               v
SentenceTransformer   BM25Okapi
      |               |
      v               v
   ChromaDB       sparse ranking
      |               |
      +-------+-------+
              |
              v
 Reciprocal Rank Fusion
              |
              v
       best policy evidence
              |
       +------+------+
       |             |
       v             v
     rag.py        agent.py
 direct RAG      tool-calling loop
                       |
          +------------+-------------+----------------+
          |            |             |                |
          v            v             v                v
      search       list topics   get policy       create demo
      policy                         page          HR request
                                                   |
                                                   v
                                                SQLite
                       |
                       v
                    app.py
                  Streamlit UI
```

---

# Main technology choices

| Layer | Tool | Why |
|---|---|---|
| Source corpus | Public GitLab Handbook Markdown | Real policy text with realistic structure and noise |
| Downloading | `requests` | Simple, transparent HTTP ingestion |
| Cleaning | Python + regex | Full control over source-specific preprocessing |
| Structural splitting | `MarkdownHeaderTextSplitter` | Preserves section hierarchy as metadata |
| Size-controlled splitting | `RecursiveCharacterTextSplitter` | Prevents oversized chunks while respecting natural boundaries |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast local embedding baseline for the English corpus |
| Vector DB | ChromaDB | Persistent local vector storage with metadata |
| Sparse retrieval | BM25 via `rank-bm25` | Strong exact-term and keyword matching |
| Rank fusion | Reciprocal Rank Fusion | Combines sparse and dense rankings without mixing incompatible scores |
| Generation | OpenAI Responses API | Reliable hosted instruction following and response generation |
| Agent actions | OpenAI function tools | Lets the model choose capabilities rather than hardcoded keyword routing |
| Persistence | SQLite | Simple local write action with no external service |
| UI | Streamlit + Plotly | Fast interactive app with chat, charts, buttons, sources, and action cards |

---

# Why use an OpenAI API model instead of a local chat model?

The project uses a **local embedding model** but a **hosted LLM for generation and agent tool calling**.

That split was deliberate.

## Local model where it makes sense

The embedding task is:

```text
text -> vector
```

For a few hundred policy chunks, a compact SentenceTransformer is fast, cheap, reproducible, and easy to run locally.

The embedding model does not need a chat template because it is not generating a conversation. It only converts text into vectors.

## Hosted LLM where it adds the most value

The generation and agent layer must:

- follow detailed policy guardrails;
- synthesize retrieved evidence;
- understand follow-up questions;
- choose tools;
- produce valid structured function arguments;
- continue after tool execution;
- write a clear final employee-facing response.

For a one-day project, a hosted model reduced time and risk related to:

- local LLM serving;
- GPU compatibility;
- model memory management;
- tokenizer chat-template differences;
- tool-calling adapters;
- local inference latency tuning;
- model-specific prompt formatting.

A future version could replace the hosted generator with a local Hugging Face instruction model. In that architecture, the tokenizer's chat template would become important.

---

# Why not GraphRAG?

GraphRAG was considered but deliberately not included in the first version.

The main project questions are policy-document questions such as:

- Can parental leave be split?
- Can coworking be reimbursed?
- How much notice is required for PTO?
- What should a new employee do during onboarding?

These are primarily retrieval and synthesis problems. A graph layer would add:

- entity extraction;
- relationship extraction;
- graph population;
- graph schema design;
- Cypher queries;
- graph traversal;
- graph/vector fusion.

That could be valuable for multi-hop questions involving complex relationships between people, roles, departments, benefits, locations, and eligibility rules. For this one-day build, hybrid retrieval plus metadata-aware RAG produced better value for the available time.

---

# Project structure

```text
peopleops-copilot/
│
├── app.py
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── storage/
│   ├── chroma/
│   └── hr_requests.db
│
├── evaluation/
│
└── src/
    ├── __init__.py
    ├── sources.py
    ├── ingest.py
    ├── cleaning.py
    ├── chunking.py
    ├── vector_store.py
    ├── hybrid_retriever.py
    ├── rag.py
    ├── database.py
    ├── tools.py
    └── agent.py
```

---

# File-by-file explanation

## `src/sources.py`

The source registry.

It contains `POLICY_SOURCES`, a list of dictionaries with fields such as:

```python
{
    "title": "Leave Types",
    "category": "leave",
    "filename": "leave_types.md",
    "page_url": "...",
    "raw_url": "...",
}
```

### Why this file exists

- ingestion does not hardcode one document;
- every policy has consistent metadata;
- new policies can be added in one place;
- the UI can list policy coverage;
- the agent can list available topics;
- source links are preserved for citations.

The corpus includes areas such as:

- Time Off Types;
- Leave Types;
- Time Away Philosophy;
- GitLab Onboarding;
- Growth and Development Fund;
- General Benefits;
- Global Travel and Expense Policy;
- Spending Company Money;
- Asynchronous Communication.

A very short Home Office Equipment page was inspected during development and replaced by a richer Travel and Expense policy source because the short page mostly redirected readers elsewhere.

---

## `src/ingest.py`

Downloads all registered policy documents into `data/raw/`.

Conceptual flow:

```text
POLICY_SOURCES
    -> loop through registry
    -> HTTP GET raw Markdown
    -> validate response
    -> save local .md file
    -> print title, path, and character count
```

This makes the corpus reproducible and refreshable.

---

## `src/cleaning.py`

Cleans raw Markdown while preserving useful semantic structure.

Typical responsibilities:

- remove YAML front matter;
- remove Hugo-specific shortcodes;
- remove selected HTML wrapper tags while preserving inner text;
- simplify Markdown links;
- remove bold and italic markers;
- normalize whitespace;
- preserve headings for structure-aware splitting.

Typical functions:

```text
remove_front_matter()
remove_hugo_shortcodes()
remove_html_wrappers()
clean_inline_markdown()
normalize_whitespace()
clean_markdown()
```

### Important design principle

> Cleaning decides what content should survive. Splitting decides how useful content should be divided.

The project does **not** aggressively lowercase or strip all punctuation before dense embedding. Natural language is useful to semantic embedding models.

---

## `src/chunking.py`

Turns cleaned Markdown into retrieval-ready documents.

Pipeline:

```text
clean Markdown
    -> split by Markdown headings
    -> preserve section/subsection/topic metadata
    -> recursively split large sections
    -> attach policy-level metadata
    -> remove heading-only / low-value fragments
    -> return one flat corpus
```

Important functions include:

```text
split_markdown_by_headers()
split_sections_into_chunks()
add_source_metadata()
is_low_value_chunk()
filter_low_value_chunks()
create_chunks()
create_all_chunks()
```

The project uses:

- `MarkdownHeaderTextSplitter`
- `RecursiveCharacterTextSplitter`

A final chunk can contain metadata such as:

```python
{
    "policy_title": "Leave Types",
    "category": "leave",
    "filename": "leave_types.md",
    "source_url": "...",
    "section": "GitLab Parental Leave",
    "subsection": "Eligibility, Entitlement and Planning Your Leave",
}
```

### Chunk-quality lesson

During QA, the smallest chunks were inspected manually.

Some tiny chunks were useless headings:

```text
### Key Features
```

But other small chunks were useful FAQ answers.

Therefore the project did **not** use a naive rule such as:

```text
delete everything under 100 characters
```

Instead, it filters low-value content more carefully.

---

## `src/vector_store.py`

Handles dense embeddings and ChromaDB.

Main functions:

```text
load_embedding_model()
get_chroma_client()
build_vector_store()
semantic_search()
```

### `load_embedding_model()`

Loads:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The function is cached with:

```python
@lru_cache(maxsize=1)
```

so a running app process reuses the same model object.

### `build_vector_store()`

- creates all chunks;
- encodes chunk text;
- stores embeddings, documents, metadata, and IDs;
- rebuilds the Chroma collection during development to avoid duplicate records.

### `semantic_search()`

- embeds the query;
- queries Chroma;
- returns top semantic matches.

The first dense test used a parental-leave question and successfully retrieved chunks containing the exact rule that leave could be taken all at once or split into up to three segments.

---

## `src/hybrid_retriever.py`

Implements the difficult retrieval layer.

Main functions:

```text
tokenize_for_bm25()
build_bm25_index()
bm25_search()
dense_search()
reciprocal_rank_fusion()
hybrid_search()
```

### Dense retrieval

Finds semantically similar content.

Useful when:

```text
Can I divide the leave?
```

needs to match:

```text
split into up to three segments
```

even though the words are not identical.

### BM25

Finds exact lexical matches and important rare terms.

Useful for terms such as:

```text
PTO
Navan
Parental Leave
coworking
```

### Reciprocal Rank Fusion

BM25 scores and vector distances are not directly comparable.

RRF therefore uses ranking positions:

```text
RRF score = sum(1 / (k + rank))
```

Conceptually:

```text
query
  |-- BM25 Top-K
  |-- dense Top-K
          |
          v
         RRF
          |
          v
      final Top-K
```

---

## `src/rag.py`

Direct RAG baseline.

Main functions:

```text
format_context()
extract_sources()
answer_question()
```

### `format_context()`

Creates structured context blocks such as:

```text
POLICY SOURCE 1

Policy: Leave Types
Section: GitLab Parental Leave
Subsection: Eligibility, Entitlement and Planning Your Leave

Policy text:
...
```

### `extract_sources()`

Creates a clean deduplicated source list for the UI.

### `answer_question()`

Runs:

```text
hybrid search
    -> context formatting
    -> source extraction
    -> model call
    -> grounded answer
```

This file remains useful as:

- a direct RAG baseline;
- a debugging path;
- an evaluation path;
- a simpler non-agent interface.

---

## `src/database.py`

Implements the local HR-request action store using SQLite.

Main functions:

```text
get_connection()
init_database()
create_hr_request()
list_hr_requests()
```

The request table stores fields such as:

```text
reference_code
created_at
topic
summary
priority
status
```

This is a real local write action, but it does **not** contact a real HR department.

---

## `src/tools.py`

Contains Python capabilities exposed to the agent.

Main tools:

```text
search_policy()
create_hr_request_tool()
list_policy_topics()
get_policy_document()
```

### Important architecture decision

`search_policy()` performs retrieval only.

It does not call the RAG LLM again.

That avoids:

```text
Agent LLM
 -> tool
 -> second LLM
 -> back to Agent LLM
```

Instead:

```text
Agent
 -> retrieval tool
 -> evidence
 -> same Agent writes grounded answer
```

---

## `src/agent.py`

The tool-calling orchestration layer.

Main parts:

```text
AGENT_INSTRUCTIONS
AGENT_TOOLS
execute_tool()
run_agent()
```

### Tool schemas

The model receives function schemas for:

- policy search;
- HR request creation;
- policy topic listing;
- original document lookup.

### `execute_tool()`

Routes a model-selected tool name to the correct Python function.

### `run_agent()`

Implements the tool loop:

```text
user request
    -> model decides whether to call a tool
    -> Python executes the tool
    -> function output is returned to the model
    -> model writes final response
```

This is intentionally different from hardcoded routing such as:

```python
if "HR" in query:
    do_something()
```

The model selects the capability through function calling.

---

## `app.py`

The Streamlit front end.

Responsibilities include:

- page configuration;
- CSS styling;
- knowledge-base metrics;
- sidebar coverage chart;
- quick prompts;
- conversation state;
- agent memory preparation;
- chat message rendering;
- source expanders;
- HR action cards;
- optional booking link.

Useful helper functions include:

```text
get_knowledge_base_stats()
render_sources()
render_actions()
render_assistant_payload()
build_agent_history()
```

---

# How to use

## 1. Clone the repository

```bash
https://github.com/MargaritaV33/peopleops-copilot.git
cd peopleops-copilot
```

## 2. Create a virtual environment

### Git Bash on Windows

```bash
python -m venv peopleops-venv
source peopleops-venv/Scripts/activate
```

### PowerShell

```powershell
python -m venv peopleops-venv
.\peopleops-venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Create `.env`

```env
OPENAI_API_KEY=your_key_here
CAL_BOOKING_URL=https://cal.com/your-booking-page
```

`CAL_BOOKING_URL` is optional.

Never commit `.env`.

## 5. Download or refresh policy sources

```bash
python -m src.ingest
```

## 6. Inspect corpus generation

```bash
python -m src.chunking
```

## 7. Build the vector store

```bash
python -c 'from src.vector_store import build_vector_store; build_vector_store()'
```

## 8. Run the application

```bash
streamlit run app.py
```

---

# What to do when policies change

The project intentionally separates source acquisition from indexing.

When source policies are updated:

```text
1. Update source URLs in src/sources.py if necessary
2. Re-download the source corpus
3. Re-run chunk inspection if desired
4. Rebuild the Chroma collection
5. Restart the Streamlit app
```

Commands:

```bash
python -m src.ingest
python -m src.chunking
python -c 'from src.vector_store import build_vector_store; build_vector_store()'
streamlit run app.py
```

The current development build recreates the collection to prevent duplicate records.

A production system could use:

- content hashes;
- document version IDs;
- incremental updates;
- deleted-document cleanup;
- scheduled refreshes;
- source-change monitoring.

---

# Example queries

## Policy questions

```text
Can I split parental leave into separate periods?
```

```text
Can I get reimbursed for using a coworking space?
```

```text
How much notice should I give before taking PTO?
```

```text
What should I focus on during onboarding?
```

## Document lookup

```text
Show me the original parental leave policy page.
```

## Capability discovery

```text
What policy areas can you help me with?
```

## Demo action

```text
I would like to request a conversation with HR about workload and wellbeing.
```

## Memory test

```text
How does parental leave work?
```

followed by:

```text
Can I split it?
```

---

# Guardrails and interpretation rules

The assistant is instructed to:

- answer policy questions only from retrieved context;
- avoid inventing eligibility, limits, procedures, or exceptions;
- clearly say when evidence is insufficient;
- treat thresholds carefully;
- distinguish between:
  - annual entitlement;
  - recommended time off;
  - consecutive leave;
  - approval thresholds;
- not present an approval threshold as a hard maximum or entitlement;
- never claim that a real HR appointment has been scheduled;
- avoid exposing internal retrieval implementation details in employee-facing answers.

---

# Current limitations

- The corpus is intentionally small and curated.
- Policy content is not refreshed automatically.
- The HR request workflow is local only.
- The optional booking button opens an external booking page; it does not create a booking through an API.
- The vector store rebuild is full rather than incremental.
- Automated evaluation is not yet comprehensive.
- Access control, authentication, and employee-specific entitlements are out of scope.
- Policy text may contain location-specific exceptions, so the assistant must not invent missing jurisdiction-specific details.

---

# Future improvements

- automated scheduled policy refresh;
- incremental Chroma updates;
- retrieval evaluation set with hit-rate metrics;
- RAGAS or TruLens evaluation;
- reranker after RRF;
- query rewriting for ambiguous follow-ups;
- source-level confidence display;
- policy-version tracking;
- authentication;
- real HR ticketing-system integration;
- real calendar integration;
- role-based or entity-based policy filtering;
- local instruction-model alternative;
- hosted deployment.

---

# Current project status

Implemented:

- [x] real policy corpus ingestion
- [x] source-specific cleaning
- [x] metadata-aware Markdown chunking
- [x] low-value chunk filtering
- [x] local embeddings
- [x] persistent ChromaDB
- [x] dense semantic retrieval
- [x] BM25 sparse retrieval
- [x] Reciprocal Rank Fusion
- [x] grounded RAG baseline
- [x] conversational context support
- [x] OpenAI function tools
- [x] SQLite HR request action
- [x] original policy document lookup
- [x] policy topic listing
- [x] Streamlit interface
- [x] source display
- [x] quick prompts
- [x] knowledge coverage chart
- [x] optional booking-link support

Not yet comprehensive:

- [ ] automated evaluation suite
- [ ] production authentication and authorization
- [ ] production HR integration
- [ ] automated policy refresh

---

## Author

**Margarita Varla**  
AI Engineering project
