"""
Pydantic Schemas for Structured Tool Inputs, Outputs, and Multi-Tool Agent Evidence.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ==========================================
# Task 1: DVC Data Management Schemas
# ==========================================

class DvcDatasetItem(BaseModel):
    name: str = Field(..., description="Dataset name or path")
    dvc_file: str = Field(..., description="DVC tracking file path (.dvc)")
    md5_hash: Optional[str] = Field(None, description="MD5 hash of dataset stored in DVC file")
    size_bytes: Optional[int] = Field(None, description="Dataset file size in bytes if available")
    is_modified: bool = Field(False, description="Whether data has changed since last commit")


class DvcDatasetProvenance(BaseModel):
    dataset_name: str
    dvc_file: str
    md5_hash: str
    git_commit: str
    branch: str
    version_tag: Optional[str] = None
    remote_storage: Optional[str] = None
    is_clean: bool = True
    evidence_summary: str


class DvcVersionDiff(BaseModel):
    dataset_name: str
    version_a: str
    version_b: str
    hash_a: Optional[str] = None
    hash_b: Optional[str] = None
    status: str = Field(..., description="IDENTICAL, MODIFIED, or UNKNOWN")
    details: str


class DvcReproductionPlan(BaseModel):
    dataset_name: str
    target_version: str
    git_commit: Optional[str] = None
    steps: List[str]
    command_to_run: str


# ==========================================
# Task 2: MLflow Experiment Intelligence Schemas
# ==========================================

class MLflowRunSummary(BaseModel):
    run_id: str
    experiment_id: str
    experiment_name: str
    status: str
    start_time: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    git_commit: Optional[str] = None
    dataset_version: Optional[str] = None
    quality_gate: Optional[str] = None


class MLflowBestCandidate(BaseModel):
    experiment_name: str
    target_metric: str
    best_run_id: str
    metric_value: float
    model_type: Optional[str] = None
    candidate_ranking: List[Dict[str, Any]]
    selection_rationale: str
    production_ready: bool
    quality_warnings: List[str] = Field(default_factory=list)


class MLflowModelRegistryDetails(BaseModel):
    model_name: str
    latest_versions: List[Dict[str, Any]] = Field(default_factory=list)
    current_production_version: Optional[str] = None
    registered_tags: Dict[str, str] = Field(default_factory=dict)


# ==========================================
# Task 3: Apache Airflow Workflow Schemas
# ==========================================

class AirflowDagItem(BaseModel):
    dag_id: str
    is_paused: bool
    is_active: bool
    description: Optional[str] = None
    timetable_description: Optional[str] = None


class AirflowDagRunStatus(BaseModel):
    dag_id: str
    dag_run_id: str
    state: str = Field(..., description="success, running, failed, queued")
    execution_date: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    conf: Dict[str, Any] = Field(default_factory=dict)


class AirflowTaskDiagnostic(BaseModel):
    dag_id: str
    dag_run_id: str
    task_id: str
    state: str
    try_number: int
    duration_seconds: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    error_message: Optional[str] = None
    failure_category: Optional[str] = None
    recommended_fix: Optional[str] = None


# ==========================================
# Task 4: Kubernetes Operations Schemas
# ==========================================

class K8sPodItem(BaseModel):
    name: str
    namespace: str
    status: str
    ready: str
    restarts: int
    age: str
    node: Optional[str] = None
    image: Optional[str] = None


class K8sDeploymentItem(BaseModel):
    name: str
    namespace: str
    replicas_desired: int
    replicas_ready: int
    replicas_available: int
    image: str
    app_version: str
    strategy: Optional[str] = None


class K8sDiagnosisReport(BaseModel):
    namespace: str
    cluster_healthy: bool
    unhealthy_pods: List[K8sPodItem] = Field(default_factory=list)
    recent_events: List[str] = Field(default_factory=list)
    root_cause_analysis: str
    recommended_resolution: str


class K8sStateChangeRequest(BaseModel):
    operation: str = Field(..., description="scale, restart, apply")
    target_resource: str
    namespace: str
    user_confirmed: bool = Field(False, description="Must be True to execute state modification")
    details: Dict[str, Any] = Field(default_factory=dict)


# ==========================================
# Task 5: Intelligent Multi-Tool Agent Schemas
# ==========================================

class ToolExecutionResult(BaseModel):
    tool_name: str
    status: str = Field("SUCCESS", description="SUCCESS, ERROR, or MOCKED")
    duration_ms: float
    data: Any
    error: Optional[str] = None


class MultiToolEvidence(BaseModel):
    query: str
    timestamp: str
    tool_results: List[ToolExecutionResult]
    known_facts: List[str]
    uncertainties: List[str]
    retraining_recommendation: bool
    retraining_rationale: str
    recommended_actions: List[str]
    synthesis_markdown: str
