"""
Data Preprocessing Stage.
Performs data validation, normalization, and train/test splitting.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Validation
    required_cols = [
        "account_age_months",
        "monthly_charges",
        "total_logins",
        "support_tickets",
        "service_failures",
        "latency_avg_ms",
        "churn",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df.isnull().sum().sum() > 0:
        print("Handling missing values (forward fill)...")
        df = df.ffill()

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    feature_names = X.columns.tolist()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Preprocessing completed. Saved {train_path} ({len(train_df)} rows) and {test_path} ({len(test_df)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess customer dataset")
    parser.add_argument("--input", default="data/raw/churn_data.csv", help="Input raw CSV path")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    preprocess_data(args.input, args.output_dir)
