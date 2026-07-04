import re


def remove_front_matter(text: str) -> str:
    """Remove YAML front matter from the beginning of a Markdown file."""

    if text.startswith("---"):
        parts = text.split("---", maxsplit=2)

        if len(parts) == 3:
            return parts[2].strip()

    return text.strip()


def remove_hugo_shortcodes(text: str) -> str:
    """Remove common Hugo shortcode markers while preserving inner text."""

    text = re.sub(r"\{\{[%<].*?[%>]\}\}", "", text)

    return text


def normalize_whitespace(text: str) -> str:
    """Reduce excessive blank lines while preserving paragraphs."""

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



# we are not removing markdown heading because we need to use them as metadata

def clean_inline_markdown(text: str) -> str:
    """Remove inline Markdown formatting while preserving readable text."""

    # Convert Markdown links:
    # [Sick Time Policy](/some/url) -> Sick Time Policy
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Remove bold Markdown:
    # **Important** -> Important
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
    )

    # Remove italic Markdown:
    # *Important text* -> Important text
    text = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"\1",
        text,
    )

    return text


def clean_markdown(text: str) -> str:
    """Apply the full Markdown-cleaning pipeline."""

    text = remove_front_matter(text)
    text = remove_hugo_shortcodes(text)
    text = remove_html_wrappers(text)
    text = clean_inline_markdown(text)
    text = normalize_whitespace(text)

    return text



from pathlib import Path


def remove_html_wrappers(text: str) -> str:
    """Remove HTML details/summary wrapper tags while preserving inner text."""

    text = re.sub(
        r"</?(?:details|summary)(?:\s[^>]*)?>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text