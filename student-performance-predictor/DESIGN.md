# Design Documentation

These diagrams satisfy the "Design & Documentation Requirements" section of
the project brief. GitHub renders Mermaid diagrams natively — view this file
on github.com to see them, or paste each block into
[mermaid.live](https://mermaid.live) to export a PNG for the PDF report.

## 1. System Architecture

```mermaid
flowchart TB
    subgraph UI["Presentation Layer"]
        A[Streamlit app.py]
    end
    subgraph Logic["Business Logic Layer"]
        B[data_manager.py]
        C[model_engine.py]
        D[analytics.py]
    end
    subgraph Persist["Persistence Layer"]
        E[(SQLite: students.db)]
        F[(models/performance_model.pkl)]
        G[[data/app.log]]
    end

    A -->|user input| B
    A -->|predict request| C
    A -->|render charts| D
    B -->|clean data| C
    C -->|load / save| F
    A -->|CRUD calls| E
    C -->|log events| G
    B -->|log events| G
```

## 2. Process / Workflow Diagram

```mermaid
flowchart LR
    Start([Student data entered]) --> Validate{Valid input?}
    Validate -- No --> ShowError[Show validation errors] --> Start
    Validate -- Yes --> Predict[Model predicts final score]
    Predict --> Categorize[Categorize into performance band]
    Categorize --> Store[Save record + prediction to SQLite]
    Store --> Display[Display result to user]
    Display --> Dashboard[Available on Analytics Dashboard]
```

## 3. Use Case Diagram

```mermaid
flowchart TB
    User((Teacher / Student))
    subgraph System["Student Performance Predictor"]
        UC1[Predict Final Score]
        UC2[Add Student Record]
        UC3[View / Update / Delete Record]
        UC4[View Analytics Dashboard]
        UC5[View Model Info]
    end
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    UC1 -.includes.-> UC2
```

## 4. Class / Component Diagram

```mermaid
classDiagram
    class ModelEngine {
        -model
        -metrics: dict
        +train(df) dict
        +predict_one(features) dict
        +feature_importance() dict
        +save(path)
        +load(path)
    }
    class DataManager {
        +generate_sample_dataset(n) DataFrame
        +load_csv(file) DataFrame
        +clean_dataset(df) DataFrame
    }
    class Database {
        +init_db()
        +add_student(record) int
        +get_all_students() list
        +update_student(id, record) bool
        +delete_student(id) bool
        +log_prediction(id, score, band)
        +get_all_predictions() list
    }
    class Analytics {
        +plot_score_distribution(df)
        +plot_correlation_heatmap(df)
        +plot_feature_importance(dict)
        +plot_predictions_over_time(df)
    }
    class StreamlitApp {
        +page_predict()
        +page_data_management()
        +page_analytics()
        +page_model_info()
    }
    StreamlitApp --> ModelEngine
    StreamlitApp --> DataManager
    StreamlitApp --> Database
    StreamlitApp --> Analytics
```

## 5. Sequence Diagram (Prediction Flow)

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit app.py
    participant V as utils.validate_student_input
    participant M as ModelEngine
    participant DB as database.py

    U->>UI: Submit student form
    UI->>V: validate_student_input(features)
    V-->>UI: errors[] (empty if valid)
    alt validation failed
        UI-->>U: Show error messages
    else validation passed
        UI->>M: predict_one(features)
        M-->>UI: {predicted_score, performance_band}
        UI->>DB: add_student(record)
        DB-->>UI: student_id
        UI->>DB: log_prediction(student_id, score, band)
        UI-->>U: Display predicted score + band
    end
```

## 6. Entity-Relationship Diagram

```mermaid
erDiagram
    STUDENTS ||--o{ PREDICTIONS : has
    STUDENTS {
        int id PK
        string name
        float study_hours_per_week
        float attendance_percentage
        float previous_grade
        float assignments_completed_pct
        float sleep_hours
        int extracurricular_activities
        int parental_support
        string created_at
    }
    PREDICTIONS {
        int id PK
        int student_id FK
        float predicted_score
        string performance_band
        string created_at
    }
```

## Dataset Description

Training uses a synthetically generated dataset (`generate_sample_dataset`
in `modules/data_manager.py`) of 800 records. The target `final_score` is
built as a weighted, noisy combination of the seven features so the
relationship is realistic but not perfectly linear — this is done so the
project is fully reproducible without depending on an external download,
while still giving both candidate models genuine signal to learn from. A
`load_csv()` function is also provided so real institutional data (matching
the same schema) can be substituted directly.
