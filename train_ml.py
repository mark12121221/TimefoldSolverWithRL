import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import json
import os

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file {csv_path} not found")
    return pd.read_csv(csv_path)

def preprocess_data(df: pd.DataFrame):
    # Features for ML
    feature_cols = [
        'num_employees', 'num_tasks', 'num_skills', 'total_required_workload',
        'total_available_capacity', 'capacity_ratio', 'avg_candidates_per_task',
        'min_candidates_per_task', 'fraction_single_candidate_tasks',
        'fraction_zero_candidate_tasks'
    ]

    X = df[feature_cols]
    y = df['is_feasible'].astype(int)

    return X, y

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name: str):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, metrics

def save_model_and_metadata(model, metrics: dict, model_path: str, metadata_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    with open(metadata_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

def main():
    csv_path = 'instances_dataset.csv'

    # Load and preprocess data
    df = load_data(csv_path)
    print(f"Loaded dataset with {len(df)} samples")
    print(f"Feasible instances: {df['is_feasible'].sum()}/{len(df)} ({df['is_feasible'].mean():.1%})")

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_trained, lr_metrics = train_and_evaluate_model(X_train, X_test, y_train, y_test, lr_model, "Logistic Regression")

    # Train Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_trained, rf_metrics = train_and_evaluate_model(X_train, X_test, y_train, y_test, rf_model, "Random Forest")

    # Save best model (Random Forest as default)
    best_model = rf_trained
    best_metrics = rf_metrics
    best_name = "random_forest"

    save_model_and_metadata(best_model, best_metrics, f'ml_artifacts/{best_name}_feasibility_model.joblib', 'ml_artifacts/model_metadata.json')

    print(f"\nBest model ({best_name}) saved with F1-score: {best_metrics['f1']:.3f}")

if __name__ == '__main__':
    main()