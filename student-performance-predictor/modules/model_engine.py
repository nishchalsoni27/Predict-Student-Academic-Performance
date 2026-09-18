"""
model_engine.py
----------------
The prediction engine. Responsible for:
  - preprocessing features into model-ready arrays
  - training and comparing candidate models (Linear Regression vs Random
    Forest) and selecting the best by cross-validated MAE
  - evaluating the chosen model
  - saving / loading the trained model with joblib
  - making single-record predictions for the Streamlit UI

Model selection rationale (see README):
  Linear Regression is included as an interpretable baseline. Random Forest
  is included because student-performance data typically has non-linear
  interactions (e.g. diminishing returns on study hours, threshold effects
  in attendance) that a linear model cannot capture. Whichever model scores
  a lower cross-validated Mean Absolute Error on the training data is kept.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from modules.utils import FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN, categorize_score, setup_logging

logger = setup_logging()


class ModelEngine:
    """Wraps a scikit-learn regressor with project-specific pre/post processing."""

    def __init__(self):
        self.model = None
        self.metrics = {}
        self.feature_names = FEATURE_COLUMNS

    # -- Preprocessing -----------------------------------------------------
    def _to_xy(self, df: pd.DataFrame):
        X = df[FEATURE_COLUMNS].values
        y = df[TARGET_COLUMN].values
        return X, y

    # -- Training ------------------------------------------------------------
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
        """Train candidate models, pick the best, evaluate on a held-out
        test split, and store the result on self.model / self.metrics."""
        X, y = self._to_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        candidates = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=random_state
            ),
        }

        cv_scores = {}
        for name, candidate in candidates.items():
            scores = cross_val_score(
                candidate, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
            )
            cv_scores[name] = -scores.mean()
            logger.info("Model %s cross-val MAE: %.3f", name, cv_scores[name])

        best_name = min(cv_scores, key=cv_scores.get)
        best_model = candidates[best_name]
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        self.metrics = {
            "selected_model": best_name,
            "cv_mae_by_model": cv_scores,
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": mean_squared_error(y_test, y_pred) ** 0.5,
            "test_r2": r2_score(y_test, y_pred),
        }
        self.model = best_model
        logger.info("Selected model: %s | Test MAE=%.2f RMSE=%.2f R2=%.3f",
                    best_name, self.metrics["test_mae"], self.metrics["test_rmse"], self.metrics["test_r2"])
        return self.metrics

    # -- Persistence -----------------------------------------------------
    def save(self, path: str = MODEL_PATH) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save. Call train() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "metrics": self.metrics}, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str = MODEL_PATH) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}. Run train_model.py first.")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.metrics = bundle["metrics"]
        logger.info("Model loaded from %s", path)

    # -- Inference -----------------------------------------------------
    def predict_one(self, features: dict) -> dict:
        """Predict a single student's final score from a feature dict."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")
        x = np.array([[features[col] for col in FEATURE_COLUMNS]])
        score = float(np.clip(self.model.predict(x)[0], 0, 100))
        return {"predicted_score": round(score, 1), "performance_band": categorize_score(score)}

    def feature_importance(self) -> dict:
        """Return feature importances if the underlying model supports it,
        else absolute coefficients (for Linear Regression) as a proxy."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            values = np.abs(self.model.coef_)
            values = values / values.sum()
        else:
            return {}
        return dict(sorted(zip(self.feature_names, values), key=lambda kv: kv[1], reverse=True))
