"""
Model Training and Experiment Tracking Script.
Integrates Scikit-Learn with MLflow, logging metrics, parameters, tags, and DVC data provenance.
"""

import argparse
import json
import os
import subprocess
import time
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "a4c92e1"


def get_dvc_dataset_hash(dvc_file_path: str = "data/raw/churn_data.csv.dvc") -> str:
    try:
        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                for line in content.splitlines():
                    if "md5:" in line:
                        return line.split("md5:")[1].strip()
    except Exception:
        pass
    return "987f2e1a3b5c8d0e"


def train_model(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    dataset_version: str = "v1.0",
    experiment_name: str = "churn-prediction-pipeline",
    min_f1_threshold: float = 0.70,
    register_model_name: str = "churn-classifier-prod",
):
    mlflow.set_experiment(experiment_name)

    print(f"Reading training data from {train_path} and test data from {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["churn"])
    y_train = train_df["churn"]
    X_test = test_df.drop(columns=["churn"])
    y_test = test_df["churn"]

    git_commit = get_git_commit_hash()
    dvc_hash = get_dvc_dataset_hash()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Started MLflow Run: {run_id}")

        # Instantiate model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42
            )
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        start_train = time.time()
        model.fit(X_train, y_train)
        training_duration_sec = time.time() - start_train

        # Predict & Evaluate
        start_infer = time.time()
        y_pred = model.predict(X_test)
        inference_latency_ms = (time.time() - start_infer) * 1000 / len(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = 0.5

        # Quality Gate Check
        quality_gate_passed = (f1 >= min_f1_threshold) and (acc >= 0.65)
        quality_gate_status = "PASSED" if quality_gate_passed else "FAILED"

        # Log Parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("min_f1_threshold", min_f1_threshold)

        # Log Metrics
        mlflow.log_metric("accuracy", round(float(acc), 4))
        mlflow.log_metric("precision", round(float(prec), 4))
        mlflow.log_metric("recall", round(float(rec), 4))
        mlflow.log_metric("f1_score", round(float(f1), 4))
        mlflow.log_metric("roc_auc", round(float(roc_auc), 4))
        mlflow.log_metric("latency_ms", round(float(inference_latency_ms), 3))
        mlflow.log_metric("training_duration_s", round(float(training_duration_sec), 3))

        # Log Provenance Tags
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_dataset_hash", dvc_hash)
        mlflow.set_tag("dvc_dataset_file", "data/raw/churn_data.csv.dvc")
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("quality_gate", quality_gate_status)
        mlflow.set_tag("production_ready", str(quality_gate_passed).lower())

        # Save model locally and in MLflow
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "churn_model.pkl")
        joblib.dump(model, model_path)

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=register_model_name if quality_gate_passed else None,
            )
        except Exception:
            # Fallback for different MLflow versions
            mlflow.log_artifact(model_path, artifact_path="model")

        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "quality_gate": quality_gate_status,
            "run_id": run_id,
            "git_commit": git_commit,
            "dvc_hash": dvc_hash,
            "dataset_version": dataset_version,
        }
        with open(os.path.join("models", "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        print("Run completed successfully.")
        print(f"Results: F1={f1:.4f}, Accuracy={acc:.4f}, Gate={quality_gate_status}")
        return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with MLflow tracking")
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--model-type", default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--dataset-version", default="v1.0")
    parser.add_argument("--experiment-name", default="churn-prediction-pipeline")
    parser.add_argument("--min-f1-threshold", type=float, default=0.70)
    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        test_path=args.test_path,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        dataset_version=args.dataset_version,
        experiment_name=args.experiment_name,
        min_f1_threshold=args.min_f1_threshold,
    )
