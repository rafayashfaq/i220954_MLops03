# System Screenshots and Exported Operational Evidence

This directory contains visual and exported evidence from the MLOps AI Agent across the 4 integrated components:

## 1. DVC Lineage & Versioning
- Versioned files: `data/raw/churn_data.csv.dvc` (MD5: `a86b256ac0f3406c13b18c7fd74d734b`)
- Version 2 files: `data/raw/churn_data_v2.csv.dvc` (MD5: `26f582fa6638ddba94c9adfcda6bfa6b`)
- DVC pipeline stages: `preprocess` and `train` in `dvc.yaml`

## 2. MLflow Tracking & Model Registry
- Experiment: `churn-prediction-pipeline`
- Model registered: `churn-classifier-prod` (Versions 1, 2, 3)
- Tracked metrics: `f1_score`, `accuracy`, `precision`, `recall`, `roc_auc`, `latency_ms`
- Quality Gate evaluation: `Gate=PASSED` and `Gate=FAILED` (injected anomaly)

## 3. Apache Airflow Pipeline
- DAG ID: `mlops_churn_training_pipeline`
- Stages: `validate_dataset -> preprocess_data -> train_model -> evaluate_and_log_mlflow -> apply_model_quality_gate`
- Retries: `retries=2`, `retry_delay=timedelta(seconds=10)`
- Failure diagnosis: `run_failed_quality_002` isolated in `apply_model_quality_gate`

## 4. Kubernetes Production Deployment
- Namespace: `mlops-prod`
- Deployment: `mlops-prediction-api` (2 replicas, liveness & readiness probes at `/health`)
- Service: `mlops-prediction-service` (ClusterIP on port 80 -> port 8000)
- ConfigMap: `mlops-prediction-config`
- Root cause diagnosis: Pod `mlops-prediction-api-broken-5d8f769b-flk32` in `CrashLoopBackOff` resolved with rollback recommendation.
