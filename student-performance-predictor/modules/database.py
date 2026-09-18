"""
database.py
-----------
SQLite persistence layer. Provides CRUD operations for two tables:

  students     -> raw student records entered by the user
  predictions  -> a log of every prediction made, for the analytics module

Kept separate from data_manager.py so the storage mechanism (SQLite) can be
swapped out later (e.g. for Postgres) without touching business logic.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from modules.utils import DB_PATH, setup_logging

logger = setup_logging()

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    study_hours_per_week REAL NOT NULL,
    attendance_percentage REAL NOT NULL,
    previous_grade REAL NOT NULL,
    assignments_completed_pct REAL NOT NULL,
    sleep_hours REAL NOT NULL,
    extracurricular_activities INTEGER NOT NULL,
    parental_support INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    predicted_score REAL NOT NULL,
    performance_band TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students (id)
);
"""


@contextmanager
def get_connection():
    """Context manager that yields a SQLite connection and guarantees
    it is closed, committing on success and rolling back on error."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database transaction failed; rolled back.")
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they do not already exist. Safe to call every run."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", DB_PATH)


# ---------------------------------------------------------------------------
# CRUD: students
# ---------------------------------------------------------------------------
def add_student(record: dict) -> int:
    """Insert a student record. Returns the new row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO students
               (name, study_hours_per_week, attendance_percentage, previous_grade,
                assignments_completed_pct, sleep_hours, extracurricular_activities,
                parental_support, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["name"],
                record["study_hours_per_week"],
                record["attendance_percentage"],
                record["previous_grade"],
                record["assignments_completed_pct"],
                record["sleep_hours"],
                int(record["extracurricular_activities"]),
                int(record["parental_support"]),
                datetime.utcnow().isoformat(),
            ),
        )
        return cur.lastrowid


def get_all_students() -> list:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


def get_student(student_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        return dict(row) if row else None


def update_student(student_id: int, record: dict) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            """UPDATE students SET
               name = ?, study_hours_per_week = ?, attendance_percentage = ?,
               previous_grade = ?, assignments_completed_pct = ?, sleep_hours = ?,
               extracurricular_activities = ?, parental_support = ?
               WHERE id = ?""",
            (
                record["name"],
                record["study_hours_per_week"],
                record["attendance_percentage"],
                record["previous_grade"],
                record["assignments_completed_pct"],
                record["sleep_hours"],
                int(record["extracurricular_activities"]),
                int(record["parental_support"]),
                student_id,
            ),
        )
        return cur.rowcount > 0


def delete_student(student_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# CRUD: predictions (append-only log, used by the analytics module)
# ---------------------------------------------------------------------------
def log_prediction(student_id: int | None, predicted_score: float, band: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO predictions (student_id, predicted_score, performance_band, created_at)
               VALUES (?, ?, ?, ?)""",
            (student_id, predicted_score, band, datetime.utcnow().isoformat()),
        )


def get_all_predictions() -> list:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]
