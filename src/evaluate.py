"""
FULL WORKING CODE (No missing data error)

Run this command:

python3 src/evaluate.py --data data/sample_data_5000.csv --out report

This creates:
- report/metrics_summary.txt
- report/metrics_summary.json
- report/plots/pr_curve.png
- report/plots/f1_comparison.png
- report/plots/detection_lead_time.png
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def classification_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    try:
        roc = roc_auc_score(y_true, y_score)
    except:
        roc = 0

    p_vals, r_vals, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(r_vals, p_vals)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


def rul_metrics(df):
    if "true_rul" not in df.columns or "pred_rul" not in df.columns:
        return {}

    df = df.dropna(subset=["true_rul", "pred_rul"])

    if len(df) == 0:
        return {}

    y_true = df["true_rul"].astype(float)
    y_pred = df["pred_rul"].astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    within_24 = np.mean(np.abs(y_true - y_pred) <= 24)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Within_24h": float(within_24),
        "Samples": int(len(df))
    }


def plot_pr_curve(y_true, y_score, save_path):
    p, r, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(r, p)
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_f1_comparison(metrics, save_path):
    names = ["Precision", "Recall", "F1 Score"]
    values = [
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"]
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("Precision vs Recall vs F1 Score")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_detection_lead_time(df, save_path):
    if "timestamp" not in df.columns:
        return

    rows = []

    for machine_id, group in df.groupby("machine_id"):
        group = group.sort_values("timestamp")

        failures = group[group["true_label"] == 1]

        for _, failure in failures.iterrows():
            previous = group[group["timestamp"] < failure["timestamp"]]
            alerts = previous[previous["pred_score"] >= 0.5]

            if len(alerts) > 0:
                last_alert = alerts.iloc[-1]

                try:
                    failure_time = pd.to_datetime(failure["timestamp"])
                    alert_time = pd.to_datetime(last_alert["timestamp"])

                    lead_hours = (
                        failure_time - alert_time
                    ).total_seconds() / 3600

                    rows.append(lead_hours)

                except:
                    pass

    if len(rows) == 0:
        return

    plt.figure(figsize=(8, 6))
    plt.hist(rows, bins=15)
    plt.title("Detection Lead Time")
    plt.xlabel("Hours")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="CSV file path"
    )

    parser.add_argument(
        "--out",
        default="report",
        help="Output folder"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print("ERROR: CSV file not found")
        print("Check path:", args.data)
        return

    df = pd.read_csv(args.data)

    required_columns = [
        "timestamp",
        "machine_id",
        "true_label",
        "pred_score"
    ]

    for col in required_columns:
        if col not in df.columns:
            print(f"ERROR: Missing column -> {col}")
            return

    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce").fillna(0).astype(int)
    df["pred_score"] = pd.to_numeric(df["pred_score"], errors="coerce").fillna(0)

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "plots"))

    class_result = classification_metrics(
        df["true_label"],
        df["pred_score"],
        args.threshold
    )

    rul_result = rul_metrics(df)

    final_result = {
        "classification": class_result,
        "rul": rul_result,
        "rows": len(df)
    }

    with open(os.path.join(args.out, "metrics_summary.json"), "w") as f:
        json.dump(final_result, f, indent=4)

    with open(os.path.join(args.out, "metrics_summary.txt"), "w") as f:
        f.write("Predictive Maintenance Evaluation Summary\n")
        f.write("=======================================\n\n")
        f.write(f"Total Rows: {len(df)}\n\n")

        f.write("CLASSIFICATION METRICS\n")
        f.write("----------------------\n")
        for k, v in class_result.items():
            f.write(f"{k}: {v}\n")

        f.write("\nRUL METRICS\n")
        f.write("----------------------\n")

        if rul_result:
            for k, v in rul_result.items():
                f.write(f"{k}: {v}\n")
        else:
            f.write("No RUL data available\n")

    plot_pr_curve(
        df["true_label"],
        df["pred_score"],
        os.path.join(args.out, "plots", "pr_curve.png")
    )

    plot_f1_comparison(
        class_result,
        os.path.join(args.out, "plots", "f1_comparison.png")
    )

    plot_detection_lead_time(
        df,
        os.path.join(args.out, "plots", "detection_lead_time.png")
    )

    print("SUCCESS: Code executed successfully")
    print("Open report folder to see graphs and F1 table")


if __name__ == "__main__":
    main()
