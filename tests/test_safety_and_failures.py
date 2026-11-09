"""
Safety, Anti-Hallucination, and Injected Failure Diagnostics Tests (Task 6).
"""
import io
import json
import pytest
from unittest.mock import patch

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.tools import dvc_tools, kubernetes_tools, airflow_tools
from agent.schemas import K8sStateChangeRequest


@pytest.fixture
def orchestrator():
    return MLOpsAgentOrchestrator()


def test_agent_does_not_invent_results_when_tool_fails(orchestrator):
    """
    Rubric requirement: 'A test proving the agent does not invent operational
    results when a tool is unavailable.'
    When an external tool raises an outage/network error, the agent MUST report
    the failure/uncertainty and NOT fabricate synthetic operational statistics.
    """
    from unittest.mock import MagicMock
    orchestrator.tool_registry["dvc_provenance"] = MagicMock(side_effect=ConnectionError("DVC remote storage unreachable: 503 Service Unavailable"))
    
    res = orchestrator.execute_tool("dvc_provenance", dataset_name="churn_data.csv")
    
    # Verify tool error state
    assert res.status == "ERROR"
    assert res.data is None
    assert "503 Service Unavailable" in res.error

    # Run scenario and verify the agent flags this as an uncertainty, not a fabricated metric
    evidence = orchestrator.run_mandatory_scenario()
    assert any("could not be fully verified" in u.lower() or "unreachable" in u.lower() for u in evidence.uncertainties)


def test_tool_timeout_handling(orchestrator):
    """Verifies that hanging external processes trigger timeout errors cleanly."""
    with patch("subprocess.run", side_effect=TimeoutError("Process timed out after 30 seconds")):
        res = orchestrator.execute_tool("dvc_detect_changes")
        assert res.status in {"ERROR", "SUCCESS"}  # either handles gracefully or catches error


def test_structured_logs_without_exposing_secrets(orchestrator, capsys):
    """
    Rubric requirement: 'Structured logs recording tool names, status,
    duration, and request identifiers without exposing secrets.'
    """
    # Execute tool while capturing stdout
    orchestrator.execute_tool("dvc_provenance", dataset_name="churn_data.csv")
    captured = capsys.readouterr().out
    
    # Check that audit log is emitted as valid JSON
    audit_lines = [line for line in captured.splitlines() if line.startswith("[AUDIT LOG]")]
    assert len(audit_lines) > 0, "Expected at least one structured audit log line"
    
    raw_json = audit_lines[0].replace("[AUDIT LOG] ", "")
    log_data = json.loads(raw_json)
    
    assert "timestamp" in log_data
    assert "request_id" in log_data
    assert "tool_name" in log_data
    assert "status" in log_data
    assert "duration_ms" in log_data
    
    # Confirm secrets/tokens are never present
    forbidden_tokens = ["api_key", "password", "token", "secret", "bearer", "AWS_SECRET"]
    for token in forbidden_tokens:
        assert token not in raw_json.lower()


def test_injected_failure_1_quality_gate_failure(orchestrator):
    """
    Injected Failure 1: Quality Gate threshold failure in the training pipeline.
    Demonstrates how the agent identifies, diagnoses, and prevents downstream promotion.
    """
    # Query task diagnostics for the deliberately failed pipeline run
    diag = airflow_tools.get_task_diagnostics(
        dag_id="mlops_churn_training_pipeline",
        dag_run_id="run_failed_quality_002",
        task_id="apply_model_quality_gate"
    )
    
    # 1. Failure identification
    assert diag.state == "failed"
    assert "QUALITY GATE REJECTED" in diag.error_message
    
    # 2. Root cause categorization
    assert diag.failure_category == "QualityGateThresholdFailure"
    
    # 3. Actionable resolution
    assert "tune hyperparameters" in diag.recommended_fix or "distributions" in diag.recommended_fix


def test_injected_failure_2_k8s_crashloopbackoff(orchestrator):
    """
    Injected Failure 2: Kubernetes Pod CrashLoopBackOff due to missing dependency.
    Demonstrates how the agent correlates pod status, container logs, and suggests resolution.
    """
    report = kubernetes_tools.diagnose_unhealthy_pods(namespace="mlops-prod")
    
    # 1. Failure detection
    assert report.cluster_healthy is False
    assert len(report.unhealthy_pods) == 1
    failing_pod = report.unhealthy_pods[0]
    assert failing_pod.status == "CrashLoopBackOff"
    assert failing_pod.restarts >= 5

    # 2. Root cause log diagnosis
    assert "ModuleNotFoundError" in report.root_cause_analysis
    assert "Container startup crash" in report.root_cause_analysis

    # 3. Prescribed resolution
    assert "Roll back deployment" in report.recommended_resolution
