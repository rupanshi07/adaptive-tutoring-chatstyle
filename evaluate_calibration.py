"""
Calibration evaluation for the live Bayesian Network.
Reads data/interaction_log.csv (real logged interactions) and computes:
1. Brier score -- mean squared error between predicted P(Correct) and
   actual correctness (0 or 1). Lower is better; 0 = perfect, 0.25 = the
   score of always predicting 50%, 1.0 = worst possible.
2. A calibration curve -- bins predictions into ranges and compares mean
   predicted probability against actual observed accuracy in that bin.

This directly measures whether the hand-specified BN probabilities are
trustworthy, not just whether they run without error.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

LOG_PATH = "data/interaction_log.csv"


def load_log():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"{LOG_PATH} not found. Run app.py and complete a few real "
            "interactions first -- this script evaluates REAL logged data, "
            "not simulated data."
        )
    df = pd.read_csv(LOG_PATH)
    df["correct_binary"] = df["correct"].astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "false": 0, "0": 0}
    )
    return df.dropna(subset=["p_correct", "correct_binary"])


def compute_brier_score(df):
    predicted = df["p_correct"].astype(float).values
    actual = df["correct_binary"].astype(float).values
    brier = np.mean((predicted - actual) ** 2)
    return brier


def compute_calibration_curve(df, num_bins=5):
    predicted = df["p_correct"].astype(float).values
    actual = df["correct_binary"].astype(float).values

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(predicted, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_mean_predicted = []
    bin_mean_actual = []
    bin_counts = []

    for b in range(num_bins):
        mask = bin_indices == b
        count = mask.sum()
        bin_counts.append(int(count))
        if count > 0:
            bin_mean_predicted.append(predicted[mask].mean())
            bin_mean_actual.append(actual[mask].mean())
        else:
            bin_mean_predicted.append(None)
            bin_mean_actual.append(None)

    return bin_edges, bin_mean_predicted, bin_mean_actual, bin_counts


def plot_calibration_curve(bin_edges, bin_mean_predicted, bin_mean_actual, bin_counts,
                             save_path="plots/calibration_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    xs = [p for p in bin_mean_predicted if p is not None]
    ys = [a for a, p in zip(bin_mean_actual, bin_mean_predicted) if p is not None]
    counts = [c for c, p in zip(bin_counts, bin_mean_predicted) if p is not None]

    if xs:
        scatter = ax.scatter(xs, ys, s=[max(c * 30, 30) for c in counts],
                              color="crimson", zorder=5, label="Observed (size = sample count)")
        for x, y, c in zip(xs, ys, counts):
            ax.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(8, 4), fontsize=8)

    ax.set_xlabel("Mean Predicted P(Correct)")
    ax.set_ylabel("Observed Accuracy")
    ax.set_title("Calibration Curve: Live Bayesian Network")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    df = load_log()
    print(f"Loaded {len(df)} real logged interactions with both prediction and outcome recorded.\n")

    brier = compute_brier_score(df)
    print(f"Brier Score: {brier:.4f}")
    print("  (0.0 = perfect, 0.25 = as good as always guessing 50%, 1.0 = worst possible)\n")

    bin_edges, bin_mean_predicted, bin_mean_actual, bin_counts = compute_calibration_curve(df)

    print("Calibration bins:")
    for i in range(len(bin_counts)):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        pred = bin_mean_predicted[i]
        actual = bin_mean_actual[i]
        n = bin_counts[i]
        if pred is not None:
            print(f"  [{lo:.1f}-{hi:.1f}] n={n}  mean predicted={pred:.2f}  observed accuracy={actual:.2f}")
        else:
            print(f"  [{lo:.1f}-{hi:.1f}] n=0  (no data in this range yet)")

    plot_calibration_curve(bin_edges, bin_mean_predicted, bin_mean_actual, bin_counts)
