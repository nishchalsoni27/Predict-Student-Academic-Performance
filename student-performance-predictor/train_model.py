"""
train_model.py
---------------
Run this once (or whenever you want to retrain) BEFORE launching app.py:

    python train_model.py

It generates the synthetic training dataset, trains + compares candidate
models, prints evaluation metrics, and saves the winning model to
models/performance_model.pkl for the Streamlit app to load.
"""

from modules.data_manager import generate_sample_dataset
from modules.model_engine import ModelEngine
from modules.utils import setup_logging

logger = setup_logging()


def main():
    logger.info("Generating training dataset...")
    df = generate_sample_dataset(n_samples=800)
    df.to_csv("data/sample_students.csv", index=False)

    logger.info("Training model...")
    engine = ModelEngine()
    metrics = engine.train(df)

    print("\n===== Training Report =====")
    print(f"Selected model : {metrics['selected_model']}")
    print(f"Test MAE       : {metrics['test_mae']:.2f}")
    print(f"Test RMSE      : {metrics['test_rmse']:.2f}")
    print(f"Test R^2       : {metrics['test_r2']:.3f}")
    print("Cross-val MAE by candidate model:")
    for name, mae in metrics["cv_mae_by_model"].items():
        print(f"  - {name}: {mae:.2f}")

    engine.save()
    print(f"\nModel saved to models/performance_model.pkl")


if __name__ == "__main__":
    main()
