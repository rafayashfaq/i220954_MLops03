"""
Unit tests for Task 2: MLflow Experiment Intelligence Agent.
"""
import pytest
from agent.tools import mlflow_tools
from agent.schemas import MLflowBestCandidate, MLflowModelRegistryDetails


def test_list_experiments_and_runs():
    exps = mlflow_tools.list_experiments()
    assert isinstance(exps, list)
    assert len(exps) > 0

    runs = mlflow_tools.list_runs(experiment_name="churn-prediction-pipeline", max_results=5)
    assert isinstance(runs, list)
    assert len(runs) > 0
    assert runs[0].run_id is not None
    assert "f1_score" in runs[0].metrics or "accuracy" in runs[0].metrics


def test_get_best_run_selection():
    best = mlflow_tools.get_best_run(experiment_name="churn-prediction-pipeline", target_metric="f1_score")
    assert isinstance(best, MLflowBestCandidate)
    assert best.best_run_id is not None
    assert best.metric_value > 0.60
    assert len(best.candidate_ranking) > 0
    # Must explain rationale and check production readiness
    assert "selected with top f1_score" in best.selection_rationale


def test_compare_runs():
    runs = mlflow_tools.list_runs(max_results=3)
    run_ids = [r.run_id for r in runs[:2]]
    comp = mlflow_tools.compare_runs(run_ids, metrics_to_compare=["f1_score", "accuracy"])
    assert comp["compared_runs_count"] == len(run_ids)
    assert len(comp["runs"]) == len(run_ids)
    for entry in comp["runs"]:
        assert "run_id" in entry
        assert "f1_score" in entry


def test_model_registry_details():
    reg = mlflow_tools.get_model_registry_details("churn-classifier-prod")
    assert isinstance(reg, MLflowModelRegistryDetails)
    assert reg.model_name == "churn-classifier-prod"
    assert len(reg.latest_versions) > 0
