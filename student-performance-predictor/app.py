"""
app.py
------
Student Performance Predictor — Streamlit entry point.

Launch with:
    streamlit run app.py

This file only handles UI orchestration and page routing. All business
logic (data, model, database, charts) lives in modules/, per the
project's modular architecture.
"""

import pandas as pd
import streamlit as st

from modules import analytics, data_manager, database
from modules.model_engine import ModelEngine
from modules.utils import FEATURE_COLUMNS, MODEL_PATH, setup_logging, validate_student_input

logger = setup_logging()

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def get_engine() -> ModelEngine:
    """Load the trained model once per session (cached across reruns)."""
    engine = ModelEngine()
    try:
        engine.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python train_model.py` in your terminal "
            "first, then restart this app."
        )
        st.stop()
    return engine


def init_state():
    database.init_db()


def page_predict(engine: ModelEngine):
    st.header("🔮 Predict Student Performance")
    st.caption("Enter a student's profile to get a predicted final score and performance band.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Student name", value="Unnamed Student")
            study_hours = st.slider("Study hours per week", 0.0, 40.0, 10.0)
            attendance = st.slider("Attendance percentage", 0.0, 100.0, 85.0)
            previous_grade = st.slider("Previous grade (0-100)", 0.0, 100.0, 70.0)
        with col2:
            assignments = st.slider("Assignments completed (%)", 0.0, 100.0, 80.0)
            sleep_hours = st.slider("Average sleep hours", 0.0, 12.0, 7.0)
            extracurricular = st.selectbox("Extracurricular activities?", ["No", "Yes"])
            parental_support = st.selectbox("Parental support level", ["Low", "Medium", "High"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        features = {
            "study_hours_per_week": study_hours,
            "attendance_percentage": attendance,
            "previous_grade": previous_grade,
            "assignments_completed_pct": assignments,
            "sleep_hours": sleep_hours,
            "extracurricular_activities": 1 if extracurricular == "Yes" else 0,
            "parental_support": {"Low": 0, "Medium": 1, "High": 2}[parental_support],
        }

        errors = validate_student_input(features)
        if errors:
            for e in errors:
                st.error(e)
            return

        try:
            result = engine.predict_one(features)
        except Exception as exc:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {exc}")
            return

        student_record = {"name": name, **features}
        student_id = database.add_student(student_record)
        database.log_prediction(student_id, result["predicted_score"], result["performance_band"])

        st.success(f"Predicted final score: **{result['predicted_score']} / 100**")
        st.info(f"Performance band: **{result['performance_band']}**")


def page_data_management():
    st.header("🗂️ Data Management")
    st.caption("Create, view, update, and delete student records stored in the database.")

    tab_view, tab_add, tab_edit = st.tabs(["View / Delete", "Add Record", "Update Record"])

    with tab_view:
        records = database.get_all_students()
        if not records:
            st.info("No student records yet. Add one in the 'Add Record' tab or make a prediction.")
        else:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
            delete_id = st.number_input("Enter student ID to delete", min_value=0, step=1)
            if st.button("Delete record"):
                if database.delete_student(int(delete_id)):
                    st.success(f"Deleted student {delete_id}.")
                    st.rerun()
                else:
                    st.warning("No record with that ID.")

    with tab_add:
        with st.form("add_record_form"):
            name = st.text_input("Name")
            study_hours = st.number_input("Study hours per week", 0.0, 80.0, 10.0)
            attendance = st.number_input("Attendance percentage", 0.0, 100.0, 85.0)
            previous_grade = st.number_input("Previous grade", 0.0, 100.0, 70.0)
            assignments = st.number_input("Assignments completed (%)", 0.0, 100.0, 80.0)
            sleep_hours = st.number_input("Sleep hours", 0.0, 24.0, 7.0)
            extracurricular = st.selectbox("Extracurricular?", ["No", "Yes"], key="add_ec")
            parental_support = st.selectbox("Parental support", ["Low", "Medium", "High"], key="add_ps")
            add_submit = st.form_submit_button("Add")

        if add_submit:
            record = {
                "name": name,
                "study_hours_per_week": study_hours,
                "attendance_percentage": attendance,
                "previous_grade": previous_grade,
                "assignments_completed_pct": assignments,
                "sleep_hours": sleep_hours,
                "extracurricular_activities": 1 if extracurricular == "Yes" else 0,
                "parental_support": {"Low": 0, "Medium": 1, "High": 2}[parental_support],
            }
            errors = validate_student_input(record)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                new_id = database.add_student(record)
                st.success(f"Added student record with ID {new_id}.")

    with tab_edit:
        records = database.get_all_students()
        if not records:
            st.info("No records to edit yet.")
        else:
            ids = [r["id"] for r in records]
            selected_id = st.selectbox("Select student ID", ids)
            current = database.get_student(selected_id)
            with st.form("edit_form"):
                name = st.text_input("Name", value=current["name"])
                study_hours = st.number_input("Study hours per week", 0.0, 80.0, float(current["study_hours_per_week"]))
                attendance = st.number_input("Attendance percentage", 0.0, 100.0, float(current["attendance_percentage"]))
                previous_grade = st.number_input("Previous grade", 0.0, 100.0, float(current["previous_grade"]))
                assignments = st.number_input("Assignments completed (%)", 0.0, 100.0, float(current["assignments_completed_pct"]))
                sleep_hours = st.number_input("Sleep hours", 0.0, 24.0, float(current["sleep_hours"]))
                update_submit = st.form_submit_button("Update")

            if update_submit:
                updated = {
                    "name": name,
                    "study_hours_per_week": study_hours,
                    "attendance_percentage": attendance,
                    "previous_grade": previous_grade,
                    "assignments_completed_pct": assignments,
                    "sleep_hours": sleep_hours,
                    "extracurricular_activities": current["extracurricular_activities"],
                    "parental_support": current["parental_support"],
                }
                database.update_student(selected_id, updated)
                st.success("Record updated.")
                st.rerun()


def page_analytics(engine: ModelEngine):
    st.header("📊 Analytics Dashboard")

    df = data_manager.generate_sample_dataset(n_samples=800)  # for population-level charts

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(analytics.plot_score_distribution(df))
    with col2:
        st.pyplot(analytics.plot_correlation_heatmap(df))

    st.subheader("Model Feature Importance")
    st.pyplot(analytics.plot_feature_importance(engine.feature_importance()))

    st.subheader("Prediction Log (this deployment's history)")
    preds = database.get_all_predictions()
    if preds:
        preds_df = pd.DataFrame(preds)
        st.dataframe(preds_df, use_container_width=True)
        st.pyplot(analytics.plot_predictions_over_time(preds_df))
    else:
        st.info("No predictions logged yet. Make one on the Predict page.")


def page_model_info(engine: ModelEngine):
    st.header("🧠 Model Information")
    st.write(f"**Selected model:** {engine.metrics.get('selected_model', 'N/A')}")
    m = engine.metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{m.get('test_mae', 0):.2f}")
    c2.metric("Test RMSE", f"{m.get('test_rmse', 0):.2f}")
    c3.metric("Test R²", f"{m.get('test_r2', 0):.3f}")

    st.write("**Cross-validated MAE by candidate model** (lower is better):")
    st.json(m.get("cv_mae_by_model", {}))

    st.write("**Features used:**")
    st.write(FEATURE_COLUMNS)


def main():
    init_state()
    engine = get_engine()

    st.sidebar.title("🎓 Student Performance Predictor")
    page = st.sidebar.radio(
        "Navigate",
        ["Predict", "Data Management", "Analytics Dashboard", "Model Info"],
    )

    if page == "Predict":
        page_predict(engine)
    elif page == "Data Management":
        page_data_management()
    elif page == "Analytics Dashboard":
        page_analytics(engine)
    elif page == "Model Info":
        page_model_info(engine)


if __name__ == "__main__":
    main()
