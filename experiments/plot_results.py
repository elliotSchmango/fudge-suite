#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    experiments_dir = Path(__file__).resolve().parent
    csv_path = experiments_dir / "fudge_results.csv"
    output_path = experiments_dir / "security_cost_curve.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = [
        "Batch_Size",
        "Epochs",
        "Privacy_Score",
        "Utility_Score",
        "Security_Score",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_columns)
    if df.empty:
        raise ValueError("No valid numeric rows found in results CSV.")

    df = df.sort_values(["Batch_Size", "Epochs"])

    plt.figure(figsize=(8, 5))
    for batch_size, group in df.groupby("Batch_Size"):
        group = group.sort_values("Epochs")
        plt.plot(
            group["Epochs"],
            group["Security_Score"],
            marker="o",
            linewidth=2,
            label=f"Batch Size {int(batch_size)}",
        )

    plt.title("Security Cost Curve")
    plt.xlabel("Unlearning Epochs")
    plt.ylabel("Security Score (ASR)")
    plt.xticks(sorted(df["Epochs"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
