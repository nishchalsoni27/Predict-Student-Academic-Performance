# Problem Statement

Teachers and academic mentors often lack an early, data-driven way to
identify students who are likely to underperform before final exams.
Manual tracking of study habits, attendance, and assignment completion is
time-consuming and inconsistent across sections.

## Scope

This project builds a supervised machine-learning system that predicts a
student's final exam score (0–100) from seven behavioral and academic
indicators, classifies that prediction into a performance band, and gives
educators a simple dashboard to manage student records and review trends.

In scope:
- Single-student prediction via a web form
- CRUD management of student records
- Model comparison (Linear Regression vs Random Forest) and automatic
  selection of the better performer
- Visual analytics: score distribution, feature correlations, feature
  importance, prediction history

Out of scope:
- Multi-class subject-wise grade prediction (only an overall final score)
- Real institutional data integration (a synthetic, realistically-correlated
  dataset is generated in-app for training; a CSV loader is provided for
  real data)
- User authentication / multi-tenant access control

## Target Users

- Teachers / academic advisors wanting an early-warning signal for at-risk students
- Students who want to see how specific habits (study time, sleep, attendance) project onto expected performance
- Course instructors evaluating this project for the Fundamentals of AI/ML course

## High-Level Features

1. **Predict** — enter a student profile, get a predicted score + performance band, logged automatically
2. **Data Management** — create, view, update, and delete stored student records (SQLite-backed CRUD)
3. **Analytics Dashboard** — population-level score distribution, correlation heatmap, model feature importance, and a running log of predictions made in the app
