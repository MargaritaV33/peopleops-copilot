from pathlib import Path
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.cleaning import clean_markdown
from src.sources import POLICY_SOURCES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


HEADERS_TO_SPLIT_ON = [
    ("##", "section"),
    ("###", "subsection"),
    ("####", "topic"),
]


def split_markdown_by_headers(markdown_text: str):
    """Split Markdown into structured sections using heading hierarchy."""

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    return header_splitter.split_text(markdown_text)


def split_sections_into_chunks(section_documents):
    """Split long sections into smaller overlapping chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return text_splitter.split_documents(section_documents)


def add_source_metadata(chunks, source: dict):
    """Attach policy-level metadata to every chunk."""

    for chunk in chunks:
        chunk.metadata.update(
            {
                "policy_title": source["title"],
                "category": source["category"],
                "source_url": source["page_url"],
                "filename": source["filename"],
            }
        )

    return chunks

def create_chunks(markdown_text: str, source: dict):
    """Run the complete cleaning and chunking pipeline."""

    cleaned_text = clean_markdown(markdown_text)

    section_documents = split_markdown_by_headers(
        cleaned_text
    )

    chunks = split_sections_into_chunks(
        section_documents
    )

    chunks = add_source_metadata(
        chunks,
        source
    )

    chunks = filter_low_value_chunks(
        chunks
    )

    return chunks



def create_all_chunks():
    """Create chunks for every policy in the source registry."""

    all_chunks = []

    for source in POLICY_SOURCES:
        raw_path = RAW_DATA_DIR / source["filename"]

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Policy file not found: {raw_path}. "
                "Run 'python -m src.ingest' first."
            )

        raw_text = raw_path.read_text(
            encoding="utf-8",
        )

        policy_chunks = create_chunks(
            markdown_text=raw_text,
            source=source,
        )

        all_chunks.extend(policy_chunks)

        print(
            f"{source['title']}: "
            f"{len(policy_chunks)} chunks"
        )

    return all_chunks

def is_low_value_chunk(chunk) -> bool:
    """Return True for chunks that contain no useful standalone context."""

    text = chunk.page_content.strip()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Example:
    # ### Key Features
    # with no body text underneath.
    header_only = bool(lines) and all(
        re.fullmatch(r"#{1,6}\s+.+", line)
        for line in lines
    )

    # Very short introductory fragments with no section context.
    unscoped_tiny_chunk = (
        len(text) < 100
        and "section" not in chunk.metadata
        and "subsection" not in chunk.metadata
        and "topic" not in chunk.metadata
    )

    return header_only or unscoped_tiny_chunk


def filter_low_value_chunks(chunks):
    """Remove chunks that provide too little retrieval value."""

    return [
        chunk
        for chunk in chunks
        if not is_low_value_chunk(chunk)
    ]


if __name__ == "__main__":
    chunks = create_all_chunks()

    # print("\n--- TINY CHUNKS UNDER 100 CHARACTERS ---")

    # for chunk in chunks:
    #     if len(chunk.page_content.strip()) < 100:
    #         print("\nLENGTH:", len(chunk.page_content.strip()))
    #         print("METADATA:", chunk.metadata)
    #         print("CONTENT:", repr(chunk.page_content))
    #         print("-" * 60)

    print("\n--- CORPUS SUMMARY ---")
    print(f"Policies: {len(POLICY_SOURCES)}")
    print(f"Total chunks: {len(chunks)}")

    chunk_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print(
        f"Smallest chunk: "
        f"{min(chunk_lengths)} characters"
    )

    print(
        f"Largest chunk: "
        f"{max(chunk_lengths)} characters"
    )

    print(
        f"Average chunk size: "
        f"{sum(chunk_lengths) / len(chunk_lengths):.1f} characters"
    )