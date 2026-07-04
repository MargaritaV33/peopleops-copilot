from pathlib import Path

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

    section_documents = split_markdown_by_headers(cleaned_text)

    chunks = split_sections_into_chunks(section_documents)

    chunks = add_source_metadata(chunks, source)

    return chunks


if __name__ == "__main__":
    source = POLICY_SOURCES[0]

    raw_path = RAW_DATA_DIR / source["filename"]

    raw_text = raw_path.read_text(encoding="utf-8")

    chunks = create_chunks(
        markdown_text=raw_text,
        source=source,
    )

    print(f"Created {len(chunks)} chunks.")

    print("\n--- FIRST 3 CHUNKS ---\n")

    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nCHUNK {index}")
        print("-" * 50)

        print("METADATA:")
        print(chunk.metadata)

        print("\nCONTENT:")
        print(chunk.page_content)

        print("\n" + "=" * 70)

    chunk_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print("\n--- CHUNK STATISTICS ---")
    print(f"Total chunks: {len(chunks)}")
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





print("\n--- CHUNK STATISTICS ---")
print(f"Total chunks: {len(chunks)}")
print(f"Smallest chunk: {min(chunk_lengths)} characters")
print(f"Largest chunk: {max(chunk_lengths)} characters")
print(
    f"Average chunk size: "
    f"{sum(chunk_lengths) / len(chunk_lengths):.1f} characters"
)