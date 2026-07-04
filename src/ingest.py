from pathlib import Path

import requests

from src.sources import POLICY_SOURCES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_policy(source: dict) -> Path:
    """Download one policy Markdown file and save it locally."""

    response = requests.get(
        source["raw_url"],
        timeout=30,
    )

    response.raise_for_status()

    output_path = RAW_DATA_DIR / source["filename"]

    output_path.write_text(
        response.text,
        encoding="utf-8",
    )

    print(f"Downloaded: {source['title']}")
    print(f"Saved to: {output_path}")
    print(f"Characters: {len(response.text):,}")

    return output_path


if __name__ == "__main__":
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source in POLICY_SOURCES:
        download_policy(source)