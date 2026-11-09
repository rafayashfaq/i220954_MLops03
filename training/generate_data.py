"""
Dataset Generator for MLOps Pipeline Demonstration.
Generates versioned synthetic customer churn / system telemetry datasets.
"""
import os
import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 1000, noise: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # Feature generation
    account_age_months = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_logins = np.random.poisson(lam=15, size=n_samples)
    support_tickets = np.random.poisson(lam=2, size=n_samples)
    service_failures = np.random.poisson(lam=0.5, size=n_samples)
    latency_avg_ms = np.random.normal(loc=150.0, scale=35.0, size=n_samples)
    
    # Churn probability logit based on features
    logit = (
        -1.5
        - 0.03 * account_age_months
        + 0.02 * monthly_charges
        - 0.05 * total_logins
        + 0.60 * support_tickets
        + 0.80 * service_failures
        + 0.01 * (latency_avg_ms - 150)
        + np.random.normal(0, noise, size=n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        "account_age_months": account_age_months,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_logins": total_logins,
        "support_tickets": support_tickets,
        "service_failures": service_failures,
        "latency_avg_ms": np.round(latency_avg_ms, 2),
        "churn": churn
    })
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    
    # Version 1: 500 samples
    df_v1 = generate_dataset(n_samples=500, random_state=42)
    df_v1.to_csv("data/raw/churn_data.csv", index=False)
    print(f"Generated data/raw/churn_data.csv (v1) with {len(df_v1)} samples.")
    
    # Version 2 baseline: 1200 samples
    df_v2 = generate_dataset(n_samples=1200, random_state=101)
    df_v2.to_csv("data/raw/churn_data_v2.csv", index=False)
    print(f"Generated data/raw/churn_data_v2.csv (v2) with {len(df_v2)} samples.")
