"""
Apache Airflow Training DAG for Production MLOps Pipeline.
Stages:
  1. validate_dataset
  2. preprocess_data
  3. train_model
  4. evaluate_and_log_mlflow
  5. apply_model_quality_gate

Features retry behavior, task dependencies, structured logging,
and halts downstream promotion if validation or quality gate fails.
"""

from datetime import datetime, timedelta
import json
import logging
import os
import sys

# Configure logger
logger = logging.getLogger("airflow.training_pipeline")
logger.setLevel(logging.INFO)

# Airflow imports with graceful fallback for standalone execution / testing
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False

    class DAG:
        def __init__(
            self,
            dag_id,
            default_args=None,
            schedule_interval=None,
            start_date=None,
            catchup=False,
            description=None,
            tags=None,
        ):
            self.dag_id = dag_id
            self.default_args = default_args or {}
            self.description = description
            self.tags = tags or []
            self.tasks = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class PythonOperator:
        def __init__(self, task_id, python_callable, op_kwargs=None, retries=0, retry_delay=None, dag=None):
            self.task_id = task_id
            self.python_callable = python_callable
            self.op_kwargs = op_kwargs or {}
            self.retries = retries
            self.retry_delay = retry_delay
            self.upstream = []
            self.downstream = []
            if dag:
                dag.tasks.append(self)

        def __rshift__(self, other):
            self.downstream.append(other)
            other.upstream.append(self)
            return other


# -------------------------------------------------------------------------
# Task Execution Callables
# -------------------------------------------------------------------------


def task_validate_dataset(**context):
    """Stage 1: Validates dataset integrity, schema, and non-empty rows."""
    logger.info("[Stage 1/5: validate_dataset] Starting dataset validation...")
    data_path = os.getenv("DATASET_PATH", "data/raw/churn_data.csv")

    if not os.path.exists(data_path):
        err = f"Dataset validation failed: File '{data_path}' not found."
        logger.error(err)
        raise FileNotFoundError(err)

    import pandas as pd

    df = pd.read_csv(data_path)
    if len(df) == 0:
        err = "Dataset validation failed: File is empty (0 rows)."
        logger.error(err)
        raise ValueError(err)

    required_cols = {"account_age_months", "monthly_charges", "total_logins", "churn"}
    missing = required_cols - set(df.columns)
    if missing:
        err = f"Dataset validation failed: Missing required columns: {missing}"
        logger.error(err)
        raise ValueError(err)

    logger.info(f"Dataset validation PASSED. Total rows: {len(df)}, Columns: {list(df.columns)}")
    return {"status": "SUCCESS", "rows": len(df), "columns": list(df.columns)}


def task_preprocess_data(**context):
    """Stage 2: Cleans, scales, and splits dataset into train/test."""
    logger.info("[Stage 2/5: preprocess_data] Preprocessing raw data...")
    from training.preprocess import preprocess_data

    input_path = os.getenv("DATASET_PATH", "data/raw/churn_data.csv")
    output_dir = "data/processed"
    preprocess_data(input_path, output_dir)
    logger.info("Preprocessing complete.")
    return {"status": "SUCCESS", "output_dir": output_dir}


def task_train_model(**context):
    """Stage 3: Trains model using scikit-learn."""
    logger.info("[Stage 3/5: train_model] Initiating model training...")
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    train_path = "data/processed/train.csv"
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["churn"])
    y_train = train_df["churn"]

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_file = "models/churn_model.pkl"
    joblib.dump(model, model_file)
    logger.info(f"Model fitted and saved to {model_file}.")
    return {"status": "SUCCESS", "model_file": model_file}


def task_evaluate_and_log_mlflow(**context):
    """Stage 4: Evaluates model metrics and logs run to MLflow."""
    logger.info("[Stage 4/5: evaluate_and_log_mlflow] Evaluating metrics and tracking run...")
    from training.train import train_model

    metrics = train_model(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        model_type="random_forest",
        n_estimators=100,
        max_depth=6,
        dataset_version="v1.0",
    )
    logger.info(
        f"MLflow Run Evaluated: F1={metrics['f1_score']}, Acc={metrics['accuracy']}, Gate={metrics['quality_gate']}"
    )
    return metrics


def task_apply_model_quality_gate(**context):
    """Stage 5: Applies deployment quality gate and prevents promotion if thresholds fail."""
    logger.info("[Stage 5/5: apply_model_quality_gate] Checking quality gate criteria...")
    metrics_path = "models/metrics.json"

    if not os.path.exists(metrics_path):
        err = "Quality gate failed: metrics.json not found."
        logger.error(err)
        raise FileNotFoundError(err)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1 = metrics.get("f1_score", 0.0)
    acc = metrics.get("accuracy", 0.0)
    min_f1 = 0.70
    min_acc = 0.75

    if f1 < min_f1 or acc < min_acc:
        err = f"QUALITY GATE REJECTED: Model F1 ({f1:.4f}) or Acc ({acc:.4f}) is below thresholds (min_f1={min_f1}, min_acc={min_acc}). Halting downstream deployment!"
        logger.error(err)
        raise ValueError(err)

    logger.info(f"QUALITY GATE APPROVED: Model passed with F1={f1:.4f}, Acc={acc:.4f}. Eligible for deployment.")
    return {"status": "APPROVED", "f1_score": f1, "accuracy": acc}


# -------------------------------------------------------------------------
# DAG Definition with Retries and Task Dependencies
# -------------------------------------------------------------------------

default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}

dag = DAG(
    dag_id="mlops_churn_training_pipeline",
    default_args=default_args,
    description="Automated ML training pipeline with DVC, MLflow, and Quality Gate verification",
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "churn", "training", "production"],
)

with dag:
    validate_dataset = PythonOperator(
        task_id="validate_dataset",
        python_callable=task_validate_dataset,
        retries=2,
        retry_delay=timedelta(seconds=5),
        dag=dag,
    )

    preprocess_data_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
        retries=2,
        retry_delay=timedelta(seconds=5),
        dag=dag,
    )

    train_model_task = PythonOperator(
        task_id="train_model", python_callable=task_train_model, retries=2, retry_delay=timedelta(seconds=5), dag=dag
    )

    evaluate_and_log_mlflow = PythonOperator(
        task_id="evaluate_and_log_mlflow",
        python_callable=task_evaluate_and_log_mlflow,
        retries=2,
        retry_delay=timedelta(seconds=5),
        dag=dag,
    )

    apply_model_quality_gate = PythonOperator(
        task_id="apply_model_quality_gate",
        python_callable=task_apply_model_quality_gate,
        retries=1,
        retry_delay=timedelta(seconds=5),
        dag=dag,
    )

    # Enforce task dependency chain:
    # validate_dataset -> preprocess_data -> train_model -> evaluate_and_log_mlflow -> apply_model_quality_gate
    validate_dataset >> preprocess_data_task >> train_model_task >> evaluate_and_log_mlflow >> apply_model_quality_gate
