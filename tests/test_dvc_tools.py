"""
Unit and safety tests for Task 1: DVC Data Management Agent.
"""

import pytest
from agent.tools import dvc_tools
from agent.schemas import DvcDatasetProvenance, DvcVersionDiff, DvcReproductionPlan


def test_list_tracked_datasets():
    datasets = dvc_tools.list_tracked_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    names = [d.name for d in datasets]
    assert any("churn_data.csv" in n for n in names)


def test_get_dataset_provenance():
    prov = dvc_tools.get_dataset_provenance("churn_data.csv")
    assert isinstance(prov, DvcDatasetProvenance)
    assert prov.dataset_name == "churn_data.csv"
    assert prov.md5_hash is not None and len(prov.md5_hash) > 5
    assert prov.git_commit is not None
    assert "churn_data.csv" in prov.evidence_summary


def test_compare_dataset_versions():
    diff_identical = dvc_tools.compare_dataset_versions("churn_data.csv", "v1.0", "v1.0")
    assert isinstance(diff_identical, DvcVersionDiff)
    assert diff_identical.status == "IDENTICAL"

    diff_mod = dvc_tools.compare_dataset_versions("churn_data.csv", "v1.0", "v2.0")
    assert diff_mod.status == "MODIFIED"
    assert "introduces altered records" in diff_mod.details


def test_explain_reproduction():
    plan = dvc_tools.explain_reproduction("churn_data.csv", "v1.0", "5246f7b")
    assert isinstance(plan, DvcReproductionPlan)
    assert len(plan.steps) == 4
    assert "git checkout 5246f7b" in plan.steps[0]
    assert "dvc repro" in plan.steps[3]


def test_dvc_command_whitelist_safety():
    """Security test: arbitrary or destructive system commands must be rejected."""
    with pytest.raises(PermissionError):
        dvc_tools._run_approved_command(["dvc", "destroy-all-data"])

    with pytest.raises(PermissionError):
        dvc_tools._run_approved_command(["rm", "-rf", "/"])
