# Production-Ready MLOps AI Agent

[![MLOps Agent CI/CD](https://github.com/rafayashfaq/i220954_MLops03/actions/workflows/ci.yml/badge.svg)](https://github.com/rafayashfaq/i220954_MLops03/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DVC Versioning](https://img.shields.io/badge/DVC-Data%20Lineage-9cf.svg)](https://dvc.org/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Cloud--Native-326CE5.svg)](https://kubernetes.io/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE.svg)](https://airflow.apache.org/)

An enterprise-grade, conversational **MLOps AI Agent** capable of understanding natural language requests, interacting with real operational infrastructure, executing authorized operations, enforcing safety boundaries, and correlating evidence across **DVC**, **MLflow**, **Apache Airflow**, and **Kubernetes**.

---

## 🌟 Key Capabilities

- **Task 1 — DVC Data Management**: Inspects versioned datasets, retrieves immutable MD5 hashes and Git commits, compares dataset versions, detects data drift, and provides step-by-step reproduction plans.
- **Task 2 — MLflow Experiment Intelligence**: Queries runs and experiments, compares evaluation metrics (F1, accuracy, latency), ranks model candidates against quality gates, and inspects model registry stages.
- **Task 3 — Apache Airflow Workflow Management**: Monitors and triggers approved training DAGs, tracks multi-stage task dependencies, isolates failed tasks, and retrieves root-cause diagnostics.
- **Task 4 — Kubernetes Cloud-Native Operations**: Observes application pods, services, and deployments; tails container logs; identifies `CrashLoopBackOff` events; and enforces strict namespace and confirmation safety gates for cluster mutations.
- **Task 5 — Intelligent Multi-Tool Reasoning**: Executes complex end-to-end operational inquiries across all four tools, distinguishes known facts from uncertainties, and strictly preserves human-in-the-loop retraining guardrails.
- **Task 6 — Reliability & Safety Controls**: Contains 33 automated tests including anti-hallucination verification when external tools fail, command whitelisting, and structured secret-masked audit logging.

---

## 📁 Repository Structure

```
assignment-03-mlops-agent/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (linting, tests, docker build)
├── agent/
│   ├── __init__.py
│   ├── main.py                  # Agent CLI, demonstration runner, and FastAPI server
│   ├── orchestrator.py          # Multi-tool agent orchestrator and reasoning engine
│   ├── schemas.py               # Pydantic schemas for structured inputs & outputs
│   └── tools/
│       ├── __init__.py
│       ├── dvc_tools.py         # Task 1: DVC tools with command whitelisting
│       ├── mlflow_tools.py      # Task 2: MLflow tools with quality-gate model ranking
│       ├── airflow_tools.py     # Task 3: Airflow workflow tools and task diagnostics
│       └── kubernetes_tools.py  # Task 4: Kubernetes observation and safety-gated mutation
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI inference microservice with /health and /predict
├── dags/
│   └── training_pipeline.py     # 5-stage Airflow training DAG with retry behavior
├── data/
│   └── raw/
│       ├── churn_data.csv       # Raw dataset
│       └── churn_data.csv.dvc   # DVC pointer file
├── kubernetes/
│   ├── deployment.yaml          # 2 replicas, probes, resource requests/limits
│   ├── service.yaml             # ClusterIP service definition
│   └── configmap.yaml           # Decoupled environment configuration
├── training/
│   ├── generate_data.py         # Synthetic telemetry & churn dataset generator
│   ├── preprocess.py            # Train/test scaling and normalization
│   └── train.py                 # Scikit-learn training script integrated with MLflow
├── tests/
│   ├── test_dvc_tools.py        # Task 1 unit & safety tests
│   ├── test_mlflow_tools.py     # Task 2 unit tests
│   ├── test_airflow_tools.py    # Task 3 workflow tests
│   ├── test_kubernetes_tools.py # Task 4 cluster inspection and mutation tests
│   ├── test_orchestrator.py     # Task 5 multi-tool integration tests
│   ├── test_safety_and_failures.py # Anti-hallucination & injected failure tests
│   └── test_api.py              # FastAPI inference service tests
├── Dockerfile                   # Multi-stage production container image
├── dvc.yaml                     # DVC pipeline stages (preprocess, train)
├── pyproject.toml               # Modern build configuration and project metadata
├── requirements.txt             # Pinned project dependencies
├── architecture.md              # Complete system architecture documentation
├── technical_report.md          # Technical report, defense evidence, and failure cases
└── README.md                    # Project manual and getting started guide
```

---

## 🚀 Quickstart & Installation

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/rafayashfaq/i220954_MLops03.git
cd i220954_MLops03

# Create and activate Python 3.11 virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# .\venv\Scripts\Activate.ps1  # On Windows PowerShell
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Running Automated Tests

Run the complete 33-test automated test suite using `pytest`:

```bash
pytest -v
```

All external Kubernetes and Airflow calls include graceful mock fallbacks so that tests execute reliably in any local or CI environment without needing live cluster connections.

Run code quality check:

```bash
ruff check agent/ api/ dags/ training/ tests/
```

---

## 🤖 Interacting with the MLOps AI Agent

### Option A: Run the 5 Required Demonstration Scenarios
Executes all 5 evaluation scenarios defined in the assignment rubric:

```bash
python agent/main.py --demo
```

**Scenarios Demonstrated**:
1. **Dataset Provenance (DVC)**: *"Which dataset version and Git commit was used for the recent model?"*
2. **Model Selection (MLflow)**: *"Compare the last five training runs and recommend the best candidate based on F1 score."*
3. **Pipeline Execution (Airflow)**: *"Trigger the model training pipeline and report whether execution succeeded."*
4. **Deployment Diagnosis (Kubernetes)**: *"Why is my prediction API unavailable? Investigate cluster health."*
5. **Multi-Tool Reasoning (Full Integration)**: *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."*

### Option B: Interactive Conversational CLI
Engage with the agent in an interactive loop:

```bash
python agent/main.py
```

Example prompt:
```
MLOps-Agent> Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.
```

### Option C: Single-Query Execution
Execute single ad-hoc questions via command-line flag:

```bash
python agent/main.py --query "Why is my prediction API unavailable?"
```

### Option D: Run Agent as an HTTP REST Service

```bash
python agent/main.py --serve --port 8080
```
- Health Check: `GET http://localhost:8080/agent/health`
- Post Natural Language Query: `POST http://localhost:8080/agent/query` with `{"query": "Compare the last five training runs"}`
- Mandatory Scenario: `GET http://localhost:8080/agent/mandatory-scenario`

---

## 🌐 Running the Inference API & Kubernetes Deployment

### Launch Inference Service Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Health Probe: `GET http://localhost:8000/health`
- Version Info: `GET http://localhost:8000/version`
- Prediction Endpoint: `POST http://localhost:8000/predict`
  ```json
  {
    "account_age_months": 24,
    "monthly_charges": 65.50,
    "total_logins": 18,
    "support_tickets": 1,
    "service_failures": 0,
    "latency_avg_ms": 140.0
  }
  ```

### Build Container Image

```bash
docker build -t mlops-prediction-api:v1.2.0 .
```

### Deploy to Kubernetes

```bash
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

---

## 🔒 Safety, Reliability & Governance Controls

1. **Subprocess Whitelisting**: DVC operations are constrained to an approved command set (`status`, `diff`, `checkout`, `pull`, `push`, `root`, `list`, `version`, `repro`). Arbitrary system shell commands are blocked.
2. **Namespace Whitelisting**: Kubernetes operations are strictly isolated to permitted namespaces (`mlops-prod`, `default`, `mlops-staging`).
3. **Mandatory Confirmation**: Any operation modifying cluster state (scale, restart, apply) requires an explicit boolean confirmation (`user_confirmed=True`).
4. **Retraining Guardrail**: Retraining is **never initiated automatically** without explicit operator consent and empirical evidence (failed quality gate or data drift).
5. **Anti-Hallucination Assurance**: If an external service is unavailable, the agent flags an uncertainty rather than synthesizing false statistics.
6. **Masked Audit Logging**: Emits structured JSON audit records with request IDs and durations while scrubbing tokens, secrets, and credentials.

---

## 📖 Further Documentation

- For deep system architecture and Mermaid sequence flows, see [architecture.md](file:///C:/Users/arraf/i220954_MLops%20A03/architecture.md).
- For the full technical report, five demonstrated scenarios with evidence, and two failure investigations, see [technical_report.md](file:///C:/Users/arraf/i220954_MLops%20A03/technical_report.md).
