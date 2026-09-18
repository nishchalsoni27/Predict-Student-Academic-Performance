"""
test_model_engine.py
---------------------
Basic unit tests. Run with:
    pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from modules.data_manager import generate_sample_dataset, clean_dataset
from modules.model_engine import ModelEngine
from modules.utils import FEATURE_COLUMNS, TARGET_COLUMN, categorize_score, validate_student_input


def test_generate_sample_dataset_shape():
    df = generate_sample_dataset(n_samples=100)
    assert len(df) == 100
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns


def test_scores_within_bounds():
    df = generate_sample_dataset(n_samples=200)
    assert df[TARGET_COLUMN].min() >= 0
    assert df[TARGET_COLUMN].max() <= 100


def test_clean_dataset_drops_nulls():
    df = generate_sample_dataset(n_samples=50)
    df.loc[0, "study_hours_per_week"] = None
    cleaned = clean_dataset(df)
    assert len(cleaned) == 49


@pytest.mark.parametrize(
    "score,expected",
    [(10, "Poor"), (50, "Average"), (70, "Good"), (95, "Excellent")],
)
def test_categorize_score(score, expected):
    assert categorize_score(score) == expected


def test_validate_student_input_flags_missing_and_out_of_range():
    errors = validate_student_input({"study_hours_per_week": 999})
    assert any("must be between" in e for e in errors)
    assert any("Missing required field" in e for e in errors)


def test_model_trains_and_predicts():
    df = generate_sample_dataset(n_samples=300)
    engine = ModelEngine()
    metrics = engine.train(df)
    assert "test_mae" in metrics
    assert metrics["test_r2"] > 0.5  # sanity check: model should learn real signal

    sample = df.iloc[0][FEATURE_COLUMNS].to_dict()
    result = engine.predict_one(sample)
    assert 0 <= result["predicted_score"] <= 100
    assert result["performance_band"] in ["Poor", "Average", "Good", "Excellent"]
