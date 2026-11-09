"""
Main Entry Point for the MLOps AI Agent.
Supports interactive conversational CLI, single-query command execution,
automated evaluation scenario demonstrations, and FastAPI serving.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Query
from pydantic import BaseModel

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import MultiToolEvidence

# Initialize orchestrator
orchestrator = MLOpsAgentOrchestrator()

# Optional FastAPI application for agent service
app = FastAPI(
    title="MLOps AI Agent Service",
    description="Conversational agent for DVC, MLflow, Airflow, and Kubernetes operations",
    version="1.0.0",
)


class AgentQueryRequest(BaseModel):
    query: str


class AgentQueryResponse(BaseModel):
    query: str
    response: str


@app.get("/agent/health")
def agent_health():
    return {"status": "UP", "tools_count": len(orchestrator.tool_registry)}


@app.post("/agent/query", response_model=AgentQueryResponse)
def query_agent(request: AgentQueryRequest):
    answer = orchestrator.route_query(request.query)
    return AgentQueryResponse(query=request.query, response=answer)


@app.get("/agent/mandatory-scenario", response_model=MultiToolEvidence)
def run_scenario():
    return orchestrator.run_mandatory_scenario()


def run_interactive_cli():
    print("=" * 70)
    print("🤖 Production MLOps AI Agent (DVC, MLflow, Airflow, Kubernetes)")
    print("Type your operational request or 'exit' to quit.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nMLOps-Agent> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Exiting MLOps Agent. Goodbye!")
                break

            response = orchestrator.route_query(user_input)
            print("\n" + response)
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


def run_demonstration_suite():
    """Executes the 5 evaluation scenarios required by the assignment rubric."""
    print("=" * 80)
    print("🚀 RUNNING 5 MLOPS EVALUATION DEMONSTRATIONS")
    print("=" * 80)

    scenarios = [
        ("Scenario 1: Dataset Provenance (DVC)", "Which dataset version and Git commit was used for the recent model?"),
        (
            "Scenario 2: Model Selection (MLflow)",
            "Compare the last five training runs and recommend the best candidate based on F1 score.",
        ),
        (
            "Scenario 3: Pipeline Execution (Airflow)",
            "Trigger the model training pipeline and report whether execution succeeded.",
        ),
        (
            "Scenario 4: Deployment Diagnosis (Kubernetes)",
            "Why is my prediction API unavailable? Investigate cluster health.",
        ),
        (
            "Scenario 5: Multi-Tool Reasoning (Full Integration)",
            "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
        ),
    ]

    for title, prompt in scenarios:
        print(f"\n🔹 {title}")
        print(f'User Prompt: "{prompt}"')
        print("-" * 70)
        output = orchestrator.route_query(prompt)
        print(output)
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="MLOps AI Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query to execute")
    parser.add_argument("--demo", action="store_true", help="Run the 5 required evaluation scenarios")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI agent HTTP service")
    parser.add_argument("--port", type=int, default=8080, help="Port for agent service")
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        print(f"Starting MLOps Agent API server on port {args.port}...")
        uvicorn.run("agent.main:app", host="0.0.0.0", port=args.port, reload=False)
    elif args.demo:
        run_demonstration_suite()
    elif args.query:
        result = orchestrator.route_query(args.query)
        print(result)
    else:
        run_interactive_cli()


if __name__ == "__main__":
    main()
