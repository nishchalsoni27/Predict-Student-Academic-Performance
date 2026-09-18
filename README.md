# Student Performance Predictor

A machine-learning powered Streamlit application that predicts a student's
final exam score from behavioral and academic indicators, and classifies it
into a performance band (Poor / Average / Good / Excellent). Built for the
Fundamentals of AI/ML course project.

## Overview

Educators often want an early read on which students are at risk before
final exams. This tool takes a small set of easily-collected indicators
(study hours, attendance, past grades, assignment completion, sleep,
extracurriculars, parental support) and predicts a numeric final score,
storing every student record and prediction for later review.

## Features

- **Predict module** — form-based input, instant score prediction + performance band, every prediction logged
- **Data Management module** — full CRUD (Create/Read/Update/Delete) on student records via SQLite
- **Analytics Dashboard module** — score distribution, feature correlation heatmap, model feature-importance chart, prediction history over time
- **Model Info page** — shows which model was selected, cross-validated MAE per candidate, and test-set metrics (MAE, RMSE, R²)
- Input validation and centralized error logging (`data/app.log`)
- Unit tests covering data generation, cleaning, validation, and model training/inference

## Technologies / Tools Used

- Python 3.10+
- Streamlit (UI)
- scikit-learn (LinearRegression, RandomForestRegressor, cross-validation)
- pandas / numpy (data handling)
- matplotlib / seaborn (visualization)
- SQLite (persistence)
- pytest (testing)

## Project Structure

```
student-performance-predictor/
├── app.py                     # Streamlit UI / page routing
├── train_model.py             # Trains & saves the model (run first)
├── modules/
│   ├── data_manager.py        # Synthetic data generation, CSV loading, cleaning
│   ├── model_engine.py        # Preprocessing, training, evaluation, prediction
│   ├── database.py            # SQLite CRUD for students & predictions
│   ├── analytics.py           # Chart-generation functions
│   └── utils.py                # Shared constants, logging, validation
├── tests/
│   └── test_model_engine.py   # Unit tests
├── data/                       # Generated dataset + SQLite DB + logs (created at runtime)
├── models/                     # Saved trained model (created at runtime)
├── requirements.txt
└── statement.md
```

## Install & Run

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd student-performance-predictor

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates models/performance_model.pkl)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Testing

```bash
pytest tests/ -v
```

## Model Selection Rationale

Linear Regression is used as an interpretable baseline; Random Forest is
included to capture potential non-linear interactions. `train_model.py`
cross-validates both (5-fold, Mean Absolute Error) and automatically keeps
whichever generalizes better — this is logged in the training report and
visible on the app's **Model Info** page.

## Evaluation Methodology

- 80/20 train/test split
- 5-fold cross-validation on the training set for model selection
- Held-out test-set metrics: MAE, RMSE, R²
- Feature importance surfaced on the Analytics Dashboard for interpretability

## Non-Functional Requirements Addressed

| Requirement | How it's met |
|---|---|
| Performance | Model cached with `st.cache_resource`; predictions return in <1s |
| Reliability | Input validation (`validate_student_input`) rejects out-of-range values before they reach the model |
| Usability | Single sidebar-navigated app, sliders/dropdowns instead of free text where possible |
| Maintainability | Modular package (`modules/`) with docstrings; each concern isolated |
| Scalability | SQLite swappable for a server DB without touching UI or model code |
| Error handling | Try/except around prediction and DB transactions, with rollback + logging |
| Logging/monitoring | All key events written to `data/app.log` via the `logging` module |

## Screenshots

<img width="1456" height="813" alt="image" src="https://github.com/user-attachments/assets/f4b1a728-8367-450b-bf6e-e5c06aeb2215" />
<img width="1456" height="818" alt="image" src="https://github.com/user-attachments/assets/71d469c9-606c-4e35-a6fb-0ba0df55e48d" />
<img width="1456" height="818" alt="image" src="https://github.com/user-attachments/assets/bfab618f-30d9-4de7-a687-5ebf9a9c5ead" />
<img width="1456" height="818" alt="image" src="https://github.com/user-attachments/assets/ac602587-6e63-4e00-9afc-388f581d2f11" />
<img width="1535" height="380" alt="image" src="https://github.com/user-attachments/assets/c6c0e051-5af7-4f84-8fc2-bd31ef5c9ec2" />



## Future Enhancements

- Support additional models (Gradient Boosting, XGBoost) in the comparison step
- Allow CSV bulk upload of student records for batch prediction
- Add authentication for multi-teacher use
- Deploy to Streamlit Community Cloud
