import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "storage" / "hr_requests.db"


def get_connection():
    """Create a connection to the local SQLite database."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sqlite3.connect(DATABASE_PATH)


def init_database():
    """Create the HR requests table if it does not already exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hr_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reference_code TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )

        connection.commit()


def create_hr_request(
    topic: str,
    summary: str,
    priority: str,
) -> dict:
    """Create and store a demo HR discussion request."""

    init_database()

    valid_priorities = {
        "low",
        "medium",
        "high",
    }

    normalized_priority = priority.lower()

    if normalized_priority not in valid_priorities:
        normalized_priority = "medium"

    reference_code = (
        f"HR-{uuid.uuid4().hex[:6].upper()}"
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    status = "Pending"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO hr_requests (
                reference_code,
                created_at,
                topic,
                summary,
                priority,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                reference_code,
                created_at,
                topic,
                summary,
                normalized_priority,
                status,
            ),
        )

        connection.commit()

    return {
        "reference_code": reference_code,
        "created_at": created_at,
        "topic": topic,
        "summary": summary,
        "priority": normalized_priority,
        "status": status,
        "message": (
            "Demo HR discussion request created successfully. "
            "This request is stored locally for demonstration purposes."
        ),
    }


def list_hr_requests(
    limit: int = 20,
) -> list[dict]:
    """Return recent demo HR requests."""

    init_database()

    with get_connection() as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                reference_code,
                created_at,
                topic,
                summary,
                priority,
                status
            FROM hr_requests
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]