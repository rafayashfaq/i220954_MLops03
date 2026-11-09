"""
Unit and safety tests for Task 3: Apache Airflow Workflow Agent.
"""
import pytest
from agent.tools import airflow_tools
from agent.schemas import AirflowDagItem, AirflowDagRunStatus, AirflowTaskDiagnostic


def test_list_dags():
    dags = airflow_tools.list_dags()
    assert isinstance(dags, list)
    assert len(dags) > 0
    dag_ids = [d.dag_id for d in dags]
    assert "mlops_churn_training_pipeline" in dag_ids


def test_get_dag_run_status():
    status = airflow_tools.get_dag_run_status("mlops_churn_training_pipeline")
    assert isinstance(status, AirflowDagRunStatus)
    assert status.dag_id == "mlops_churn_training_pipeline"
    assert status.state in {"success", "running", "failed", "queued"}


def test_trigger_approved_dag():
    res = airflow_tools.trigger_training_dag("mlops_churn_training_pipeline")
    assert isinstance(res, AirflowDagRunStatus)
    assert res.dag_id == "mlops_churn_training_pipeline"
    assert res.state in {"running", "queued", "success"}


def test_trigger_unapproved_dag_rejected():
    """Security test: triggering arbitrary unapproved DAGs must be rejected."""
    with pytest.raises(PermissionError) as exc_info:
        airflow_tools.trigger_training_dag("malicious_unapproved_dag")
    assert "Security Policy Violation" in str(exc_info.value)


def test_get_task_diagnostics():
    diag = airflow_tools.get_task_diagnostics(
        dag_id="mlops_churn_training_pipeline",
        dag_run_id="run_failed_quality_002",
        task_id="apply_model_quality_gate"
    )
    assert isinstance(diag, AirflowTaskDiagnostic)
    assert diag.task_id == "apply_model_quality_gate"
    assert diag.state == "failed"
    assert "QUALITY GATE REJECTED" in diag.error_message
    assert diag.recommended_fix is not None
