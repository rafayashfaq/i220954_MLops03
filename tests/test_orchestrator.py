"""
Integration tests for Task 5: Multi-Tool Agent Orchestrator.
"""
import pytest
from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import MultiToolEvidence


@pytest.fixture
def orchestrator():
    return MLOpsAgentOrchestrator()


def test_tool_registry_and_duration_tracking(orchestrator):
    res = orchestrator.execute_tool("dvc_provenance", dataset_name="churn_data.csv")
    assert res.status == "SUCCESS"
    assert res.tool_name == "dvc_provenance"
    assert res.duration_ms >= 0.0
    assert res.data is not None


def test_unregistered_tool_handling(orchestrator):
    res = orchestrator.execute_tool("non_existent_tool_action")
    assert res.status == "ERROR"
    assert "not registered" in res.error


def test_mandatory_integration_scenario(orchestrator):
    """
    Validates mandatory integration scenario:
    User: 'Find the best recent model, identify the dataset version used to train it,
    check the deployment status, and explain whether retraining is necessary.'
    """
    evidence = orchestrator.run_mandatory_scenario()
    assert isinstance(evidence, MultiToolEvidence)
    
    # 1. Multi-tool results collected
    assert len(evidence.tool_results) >= 4
    tool_names = [tr.tool_name for tr in evidence.tool_results]
    assert "mlflow_best_candidate" in tool_names
    assert "dvc_provenance" in tool_names
    assert "k8s_deployed_version" in tool_names
    assert "airflow_dag_run_status" in tool_names

    # 2. Distinguishes known facts vs uncertainties
    assert len(evidence.known_facts) > 0
    assert any("MLflow" in f or "model" in f for f in evidence.known_facts)
    assert any("Kubernetes" in f or "deployment" in f for f in evidence.known_facts)

    # 3. Retraining guardrail: Retraining is never initiated automatically
    cleaned_md = evidence.synthesis_markdown.lower().replace("*", "")
    assert "not triggered automatically" in cleaned_md or "not initiated automatically" in cleaned_md
    assert evidence.retraining_rationale is not None


def test_natural_language_query_routing(orchestrator):
    # DVC provenance routing
    ans_dvc = orchestrator.route_query("Which dataset version was used for recent experiment?")
    assert "DVC Dataset Provenance" in ans_dvc

    # MLflow comparison routing
    ans_ml = orchestrator.route_query("Compare the last five training runs and recommend best model based on F1 score")
    assert "MLflow Model Comparison" in ans_ml or "Candidate Ranking" in ans_ml

    # Kubernetes diagnosis routing
    ans_k8s = orchestrator.route_query("Why is my prediction API unavailable?")
    assert "Kubernetes Health Diagnosis" in ans_k8s
    assert "Root Cause" in ans_k8s

    # Airflow trigger routing
    ans_af = orchestrator.route_query("Trigger the model training pipeline")
    assert "Triggered Apache Airflow Pipeline" in ans_af
