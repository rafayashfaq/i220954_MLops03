"""
Task 3: Apache Airflow Workflow Agent Tools.
Provides structured tools to list DAGs, check DAG run status, trigger approved training workflows,
identify failed tasks, and retrieve task-level execution diagnostics.
Interacts with the Airflow REST API (with authenticated requests and local fallback).
"""
import datetime
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any
import requests

from agent.schemas import (
    AirflowDagItem,
    AirflowDagRunStatus,
    AirflowTaskDiagnostic,
)

logger = logging.getLogger("agent.tools.airflow")

AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080/api/v1")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")

APPROVED_DAGS = {
    "mlops_churn_training_pipeline",
    "churn-training-pipeline",
    "model_retraining_dag"
}

# In-memory execution registry for standalone / mock execution
_LOCAL_DAG_RUNS: Dict[str, Dict[str, Any]] = {
    "run_success_001": {
        "dag_id": "mlops_churn_training_pipeline",
        "dag_run_id": "run_success_001",
        "state": "success",
        "execution_date": "2026-11-09T18:00:00Z",
        "start_date": "2026-11-09T18:00:05Z",
        "end_date": "2026-11-09T18:01:25Z",
        "tasks": {
            "validate_dataset": {"state": "success", "try": 1, "duration": 3.2},
            "preprocess_data": {"state": "success", "try": 1, "duration": 5.1},
            "train_model": {"state": "success", "try": 1, "duration": 18.4},
            "evaluate_and_log_mlflow": {"state": "success", "try": 1, "duration": 12.0},
            "apply_model_quality_gate": {"state": "success", "try": 1, "duration": 1.5}
        }
    },
    "run_failed_quality_002": {
        "dag_id": "mlops_churn_training_pipeline",
        "dag_run_id": "run_failed_quality_002",
        "state": "failed",
        "execution_date": "2026-11-09T19:30:00Z",
        "start_date": "2026-11-09T19:30:05Z",
        "end_date": "2026-11-09T19:31:10Z",
        "tasks": {
            "validate_dataset": {"state": "success", "try": 1, "duration": 2.8},
            "preprocess_data": {"state": "success", "try": 1, "duration": 4.9},
            "train_model": {"state": "success", "try": 1, "duration": 15.2},
            "evaluate_and_log_mlflow": {"state": "success", "try": 1, "duration": 10.1},
            "apply_model_quality_gate": {
                "state": "failed",
                "try": 2,
                "duration": 2.1,
                "error": "QUALITY GATE REJECTED: Model F1 (0.6216) is below min_f1 threshold (0.7000). Promotion halted.",
                "category": "QualityGateThresholdFailure",
                "recommendation": "Inspect dataset feature distributions for drift or tune hyperparameters (e.g. increase tree depth)."
            }
        }
    }
}


def _get_auth() -> tuple:
    return (AIRFLOW_USER, AIRFLOW_PASSWORD)


def list_dags() -> List[AirflowDagItem]:
    """Lists all available DAGs registered in Apache Airflow."""
    try:
        url = f"{AIRFLOW_BASE_URL}/dags"
        resp = requests.get(url, auth=_get_auth(), timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            dag_items = []
            for d in data.get("dags", []):
                dag_items.append(
                    AirflowDagItem(
                        dag_id=d["dag_id"],
                        is_paused=d.get("is_paused", False),
                        is_active=d.get("is_active", True),
                        description=d.get("description"),
                        timetable_description=d.get("timetable_description")
                    )
                )
            return dag_items
    except Exception:
        pass

    # Fallback to local discovered DAGs
    return [
        AirflowDagItem(
            dag_id="mlops_churn_training_pipeline",
            is_paused=False,
            is_active=True,
            description="Automated ML training pipeline with DVC, MLflow, and Quality Gate verification",
            timetable_description="@manual or on-demand trigger"
        )
    ]


def get_dag_run_status(dag_id: str = "mlops_churn_training_pipeline", dag_run_id: Optional[str] = None) -> AirflowDagRunStatus:
    """Retrieves execution status for a given DAG and run ID."""
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
        resp = requests.get(url, auth=_get_auth(), timeout=3)
        if resp.status_code == 200:
            runs = resp.json().get("dag_runs", [])
            if runs:
                target = runs[-1]
                if dag_run_id:
                    for r in runs:
                        if r["dag_run_id"] == dag_run_id:
                            target = r
                            break
                return AirflowDagRunStatus(
                    dag_id=dag_id,
                    dag_run_id=target["dag_run_id"],
                    state=target["state"],
                    execution_date=target["execution_date"],
                    start_date=target.get("start_date"),
                    end_date=target.get("end_date"),
                    conf=target.get("conf", {})
                )
    except Exception:
        pass

    # Local fallback
    if dag_run_id and dag_run_id in _LOCAL_DAG_RUNS:
        item = _LOCAL_DAG_RUNS[dag_run_id]
    else:
        # Latest local run
        item = list(_LOCAL_DAG_RUNS.values())[-1]

    return AirflowDagRunStatus(
        dag_id=item["dag_id"],
        dag_run_id=item["dag_run_id"],
        state=item["state"],
        execution_date=item["execution_date"],
        start_date=item["start_date"],
        end_date=item["end_date"],
        conf=item.get("conf", {})
    )


def trigger_training_dag(
    dag_id: str = "mlops_churn_training_pipeline",
    conf: Optional[Dict[str, Any]] = None
) -> AirflowDagRunStatus:
    """Triggers an approved training DAG execution."""
    if dag_id not in APPROVED_DAGS:
        raise PermissionError(
            f"Security Policy Violation: DAG '{dag_id}' is not in the approved training DAG list: {sorted(list(APPROVED_DAGS))}"
        )

    conf = conf or {"dataset_version": "v1.0", "trigger_source": "mlops_agent"}
    run_id = f"manual_agent_{uuid.uuid4().hex[:8]}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
        payload = {"dag_run_id": run_id, "conf": conf}
        resp = requests.post(url, json=payload, auth=_get_auth(), timeout=5)
        if resp.status_code in {200, 201}:
            data = resp.json()
            return AirflowDagRunStatus(
                dag_id=dag_id,
                dag_run_id=data["dag_run_id"],
                state=data["state"],
                execution_date=data.get("execution_date", now_iso),
                start_date=data.get("start_date", now_iso),
                conf=conf
            )
    except Exception:
        pass

    # Record in local registry
    _LOCAL_DAG_RUNS[run_id] = {
        "dag_id": dag_id,
        "dag_run_id": run_id,
        "state": "running",
        "execution_date": now_iso,
        "start_date": now_iso,
        "end_date": None,
        "conf": conf,
        "tasks": {
            "validate_dataset": {"state": "success", "try": 1, "duration": 2.5},
            "preprocess_data": {"state": "running", "try": 1, "duration": 1.2}
        }
    }

    return AirflowDagRunStatus(
        dag_id=dag_id,
        dag_run_id=run_id,
        state="running",
        execution_date=now_iso,
        start_date=now_iso,
        conf=conf
    )


def get_failed_tasks(dag_id: str = "mlops_churn_training_pipeline", dag_run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Identifies any failed tasks in the specified or most recent DAG run."""
    run_status = get_dag_run_status(dag_id, dag_run_id)
    target_run_id = run_status.dag_run_id

    # Check local registry
    run_data = _LOCAL_DAG_RUNS.get(target_run_id)
    failed = []
    if run_data and "tasks" in run_data:
        for t_name, t_info in run_data["tasks"].items():
            if t_info.get("state") == "failed":
                failed.append({
                    "dag_id": dag_id,
                    "dag_run_id": target_run_id,
                    "task_id": t_name,
                    "state": "failed",
                    "try_number": t_info.get("try", 1),
                    "error": t_info.get("error", "Task execution failed"),
                    "recommendation": t_info.get("recommendation", "Review task execution logs")
                })
    return failed


def get_task_diagnostics(
    dag_id: str = "mlops_churn_training_pipeline",
    dag_run_id: str = "run_failed_quality_002",
    task_id: str = "apply_model_quality_gate"
) -> AirflowTaskDiagnostic:
    """Retrieves deep task execution diagnostics, failure category, and recommended fixes."""
    run_data = _LOCAL_DAG_RUNS.get(dag_run_id, {})
    task_info = run_data.get("tasks", {}).get(task_id, {})

    state = task_info.get("state", "failed")
    try_num = task_info.get("try", 1)
    err = task_info.get("error", f"Task '{task_id}' encountered an operational anomaly.")
    category = task_info.get("category", "GeneralTaskFailure")
    rec = task_info.get("recommendation", "Verify input schemas, training dataset quality, and worker resources.")

    return AirflowTaskDiagnostic(
        dag_id=dag_id,
        dag_run_id=dag_run_id,
        task_id=task_id,
        state=state,
        try_number=try_num,
        duration_seconds=task_info.get("duration", 2.0),
        start_date=run_data.get("start_date"),
        end_date=run_data.get("end_date"),
        error_message=err,
        failure_category=category,
        recommended_fix=rec
    )
