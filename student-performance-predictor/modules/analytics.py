"""
analytics.py
------------
Reporting/visualization module. Every function returns a matplotlib Figure
so app.py can render it with st.pyplot() without this module importing
Streamlit directly (keeps it independently testable).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modules.utils import FEATURE_COLUMNS, TARGET_COLUMN


def plot_score_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[TARGET_COLUMN], bins=20, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of Final Scores")
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(importance.keys())
    values = list(importance.values())
    sns.barplot(x=values, y=names, ax=ax, color="#55A868")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Relative Importance")
    fig.tight_layout()
    return fig


def plot_predictions_over_time(predictions_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    predictions_df = predictions_df.sort_values("created_at")
    ax.plot(range(len(predictions_df)), predictions_df["predicted_score"], marker="o")
    ax.set_title("Predicted Scores Over Time (this session's log)")
    ax.set_xlabel("Prediction #")
    ax.set_ylabel("Predicted Score")
    fig.tight_layout()
    return fig
