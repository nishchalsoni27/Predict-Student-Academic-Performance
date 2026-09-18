"""
utils.py
--------
Shared constants, logging setup, and small helper functions used across
the Student Performance Predictor project.
"""

import logging
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "study_hours_per_week",
    "attendance_percentage",
    "previous_grade",
    "assignments_completed_pct",
    "sleep_hours",
    "extracurricular_activities",  # 0 = No, 1 = Yes
    "parental_support",            # 0 = Low, 1 = Medium, 2 = High
]

TARGET_COLUMN = "final_score"

PERFORMANCE_BANDS = [
    (0, 40, "Poor"),
    (40, 60, "Average"),
    (60, 80, "Good"),
    (80, 101, "Excellent"),
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "performance_model.pkl")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "students.db")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "app.log")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure and return a module-level logger that writes to a file
    and to stdout. Called once at application startup."""
    logger = logging.getLogger("spp")
    if logger.handlers:  # avoid duplicate handlers on Streamlit re-runs
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def categorize_score(score: float) -> str:
    """Map a numeric final score (0-100) to a human-readable performance band."""
    for low, high, label in PERFORMANCE_BANDS:
        if low <= score < high:
            return label
    return "Unknown"


def validate_student_input(data: dict) -> list:
    """Validate a single student's input dict against expected ranges.
    Returns a list of error messages (empty list == valid)."""
    errors = []

    def in_range(key, low, high):
        if key in data and not (low <= data[key] <= high):
            errors.append(f"{key} must be between {low} and {high} (got {data[key]}).")

    in_range("study_hours_per_week", 0, 80)
    in_range("attendance_percentage", 0, 100)
    in_range("previous_grade", 0, 100)
    in_range("assignments_completed_pct", 0, 100)
    in_range("sleep_hours", 0, 24)

    for key in FEATURE_COLUMNS:
        if key not in data:
            errors.append(f"Missing required field: {key}")

    return errors
