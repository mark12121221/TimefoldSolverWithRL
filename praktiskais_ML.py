from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Configuration
DATASET_PATH = "instances_dataset_timefold_unique_3000.csv"
MODEL_DIR = Path("ml_artifacts")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "is_feasible"

FEATURE_COLUMNS = [
    "num_employees",
    "num_tasks",
    "num_skills",
    "total_required_workload",
    "total_available_capacity",
    "capacity_ratio",
    "avg_candidates_per_task",
    "min_candidates_per_task",
    "fraction_single_candidate_tasks",
    "fraction_zero_candidate_tasks",
]

RANDOM_STATE = 42


# Data loading
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"There are missing columns in the dataset: {missing_cols}")

    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


# Preprocessing
def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
        ]
    )
    return preprocessor


# Models
def build_logistic_regression_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def build_random_forest_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def build_mlp_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


# Evaluation
def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_proba = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def print_metrics(model_name: str, metrics: Dict[str, Any]) -> None:
    print(f"\n=== {model_name} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    if "roc_auc" in metrics:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(metrics["classification_report"])


# Training
def train_and_select_best_model(dataset_path: str) -> None:
    df = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df)

    candidates = {
        "logistic_regression": build_logistic_regression_pipeline(),
        "random_forest": build_random_forest_pipeline(),
        "mlp_neural_network": build_mlp_pipeline(),
    }

    results = {}
    trained_models = {}

    for name, model in candidates.items():
        print(f"\nTraining model: {name}")
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)

        results[name] = metrics
        trained_models[name] = model

        print_metrics(name, metrics)

    best_name = max(results.keys(), key=lambda n: results[n]["f1"])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    print(f"\nBest model: {best_name}")

    model_path = MODEL_DIR / f"{best_name}_feasibility_model.joblib"
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": best_metrics,
        "all_results": results,
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")


# Inference
def load_trained_model(model_path: str) -> Pipeline:
    return joblib.load(model_path)


def predict_instance_feasibility(model: Pipeline, instance_features: Dict[str, float]) -> Dict[str, Any]:
    row = pd.DataFrame([instance_features], columns=FEATURE_COLUMNS)

    prediction = int(model.predict(row)[0])

    result = {
        "predicted_is_feasible": prediction,
    }

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0, 1])
        result["feasible_probability"] = probability

    return result


# Example usage
if __name__ == "__main__":
    train_and_select_best_model(DATASET_PATH)

    saved_model_path = next(MODEL_DIR.glob("*_feasibility_model.joblib"))
    model = load_trained_model(str(saved_model_path))

    new_instance_features = {
        "num_employees": 15,
        "num_tasks": 40,
        "num_skills": 5,
        "total_required_workload": 300,
        "total_available_capacity": 420,
        "capacity_ratio": 300 / 420,
        "avg_candidates_per_task": 3.2,
        "min_candidates_per_task": 1,
        "fraction_single_candidate_tasks": 0.15,
        "fraction_zero_candidate_tasks": 0.02,
    }

    prediction = predict_instance_feasibility(model, new_instance_features)
    print("\nPrediction for new instance:")
    print(prediction)