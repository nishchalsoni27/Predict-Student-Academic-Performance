"""
data_manager.py
----------------
Handles all data *input* concerns: generating a synthetic training dataset
(so the project is fully self-contained and does not depend on an external
download), loading a user-supplied CSV, and basic cleaning/validation.

This module is deliberately separate from database.py (persisted student
records) and model_engine.py (ML logic) to keep responsibilities modular.
"""

import numpy as np
import pandas as pd

from modules.utils import FEATURE_COLUMNS, TARGET_COLUMN, setup_logging

logger = setup_logging()


def generate_sample_dataset(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic but realistically-correlated dataset for training.

    The target (final_score) is built as a weighted combination of the
    features plus noise, so the model has genuine signal to learn from.
    """
    rng = np.random.default_rng(random_state)

    study_hours = rng.uniform(0, 40, n_samples)
    attendance = rng.uniform(40, 100, n_samples)
    previous_grade = rng.uniform(30, 100, n_samples)
    assignments = rng.uniform(0, 100, n_samples)
    sleep_hours = rng.normal(7, 1.5, n_samples).clip(3, 12)
    extracurricular = rng.integers(0, 2, n_samples)
    parental_support = rng.integers(0, 3, n_samples)

    noise = rng.normal(0, 6, n_samples)

    final_score = (
        0.28 * study_hours
        + 0.22 * attendance
        + 0.30 * previous_grade
        + 0.12 * assignments
        + 2.0 * (sleep_hours - 7).clip(-3, 3) * -1  # too little/too much sleep hurts slightly
        + 2.5 * extracurricular
        + 3.0 * parental_support
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "study_hours_per_week": study_hours.round(1),
            "attendance_percentage": attendance.round(1),
            "previous_grade": previous_grade.round(1),
            "assignments_completed_pct": assignments.round(1),
            "sleep_hours": sleep_hours.round(1),
            "extracurricular_activities": extracurricular,
            "parental_support": parental_support,
            "final_score": final_score.round(1),
        }
    )
    logger.info("Generated synthetic dataset with %d rows.", n_samples)
    return df


def load_csv(file) -> pd.DataFrame:
    """Load a user-uploaded CSV and verify it has the required columns."""
    df = pd.read_csv(file)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {sorted(missing)}")
    return clean_dataset(df)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with nulls in required columns and clip out-of-range values."""
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing values during cleaning.", dropped)

    df["attendance_percentage"] = df["attendance_percentage"].clip(0, 100)
    df["previous_grade"] = df["previous_grade"].clip(0, 100)
    df["assignments_completed_pct"] = df["assignments_completed_pct"].clip(0, 100)
    df["sleep_hours"] = df["sleep_hours"].clip(0, 24)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(0, 100)
    return df
