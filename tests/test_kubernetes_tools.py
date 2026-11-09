"""
Unit and safety tests for Task 4: Kubernetes Operations Agent.
"""
import pytest
from agent.tools import kubernetes_tools
from agent.schemas import K8sDiagnosisReport, K8sStateChangeRequest


def test_list_pods_and_deployments():
    pods = kubernetes_tools.list_pods(namespace="mlops-prod")
    assert isinstance(pods, list)
    assert len(pods) >= 2
    assert all(p.namespace == "mlops-prod" for p in pods)

    deployments = kubernetes_tools.list_deployments(namespace="mlops-prod")
    assert isinstance(deployments, list)
    assert len(deployments) > 0
    assert deployments[0].replicas_desired >= 2


def test_inspect_service():
    svc = kubernetes_tools.inspect_service("mlops-prediction-service", namespace="mlops-prod")
    assert svc["type"] == "ClusterIP"
    assert svc["ports"][0]["port"] == 80
    assert svc["ports"][0]["target_port"] == 8000


def test_diagnose_unhealthy_pods():
    diag = kubernetes_tools.diagnose_unhealthy_pods(namespace="mlops-prod")
    assert isinstance(diag, K8sDiagnosisReport)
    assert diag.namespace == "mlops-prod"
    # Detects the mock broken pod and root cause
    assert len(diag.unhealthy_pods) > 0
    assert "CrashLoopBackOff" in diag.unhealthy_pods[0].status
    assert "ModuleNotFoundError" in diag.root_cause_analysis


def test_unauthorized_namespace_access_rejected():
    """Security test: access to unauthorized namespace must be denied."""
    with pytest.raises(PermissionError):
        kubernetes_tools.list_pods(namespace="kube-system-classified")


def test_unconfirmed_cluster_modification_rejected():
    """Safety requirement: Any operation changing cluster state must require explicit confirmation."""
    req = K8sStateChangeRequest(
        operation="scale",
        target_resource="deployment/mlops-prediction-api",
        namespace="mlops-prod",
        user_confirmed=False  # Missing confirmation!
    )
    with pytest.raises(PermissionError) as exc_info:
        kubernetes_tools.execute_cluster_state_change(req)
    assert "SAFETY VIOLATION" in str(exc_info.value)
    assert "user_confirmed" in str(exc_info.value)


def test_confirmed_cluster_modification_allowed():
    """Confirmed operation in approved namespace executes successfully."""
    req = K8sStateChangeRequest(
        operation="restart",
        target_resource="deployment/mlops-prediction-api",
        namespace="mlops-prod",
        user_confirmed=True
    )
    res = kubernetes_tools.execute_cluster_state_change(req)
    assert res["status"] == "EXECUTED"
    assert res["operation"] == "restart"
