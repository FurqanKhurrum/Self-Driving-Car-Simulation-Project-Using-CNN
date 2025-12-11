# scripts/explore_data.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import balance_steering_data

DATA_DIR = "data"
RUN_FOLDERS = [
    "run1_clean",
    "run2_reverse",
    "run3_recovery",
]


def load_run(run_name):
    """
    Load a single run's driving_log.csv as a DataFrame and
    add a 'run' column.
    """
    csv_path = os.path.join(DATA_DIR, run_name, "driving_log.csv")

    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found, skipping.")
        return None

    df = pd.read_csv(csv_path, header=None)
    # Standard Udacity format:
    # 0:center, 1:left, 2:right, 3:steering, 4:throttle, 5:brake, 6:speed
    df.columns = [
        "center",
        "left",
        "right",
        "steering",
        "throttle",
        "brake",
        "speed",
    ]
    df["run"] = run_name
    return df


def merge_runs(run_names):
    """
    Load all runs, concatenate into a single DataFrame.
    """
    dfs = []
    for r in run_names:
        df = load_run(r)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid runs found. Check RUN_FOLDERS paths.")

    merged = pd.concat(dfs, ignore_index=True)
    return merged


def save_merged_csv(df, out_path="data/merged_driving_log.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Merged CSV saved to: {out_path}")


def show_basic_stats(df):
    print("\n[INFO] Basic info:")
    print(df.info())

    print("\n[INFO] First 5 rows:")
    print(df.head())

    print("\n[INFO] Steering angle stats (overall):")
    print(df["steering"].describe())

    print("\n[INFO] Samples per run:")
    print(df["run"].value_counts())


def plot_steering_histograms(df):
    # Overall distribution
    plt.figure(figsize=(8, 4))
    df["steering"].hist(bins=31)
    plt.title("Steering Angle Distribution (All Runs)")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    # Per-run distributions
    runs = df["run"].unique()
    n_runs = len(runs)

    plt.figure(figsize=(4 * n_runs, 3))
    for i, run_name in enumerate(runs, start=1):
        ax = plt.subplot(1, n_runs, i)
        sub = df[df["run"] == run_name]
        sub["steering"].hist(bins=31, ax=ax)
        ax.set_title(run_name)
        ax.set_xlabel("steering")
        ax.set_ylabel("count")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("[INFO] Loading runs:", RUN_FOLDERS)
    merged_df = merge_runs(RUN_FOLDERS)

    # Save original merged CSV (optional)
    merged_df.to_csv("data/merged_driving_log_raw.csv", index=False)

    show_basic_stats(merged_df)
    plot_steering_histograms(merged_df)  # per-run BEFORE balancing

    # ------------ BALANCE DATA HERE ------------
    balanced_df = balance_steering_data(
        merged_df,
        n_bins=31,
        samples_per_bin=700,   # you can tweak this
        display=True           # shows before/after overall hist
    )

    # Save balanced CSV
    balanced_df.to_csv("data/merged_driving_log_balanced.csv", index=False)

    print("\n[INFO] Stats AFTER balancing:")
    show_basic_stats(balanced_df)

    print("\n[INFO] Per-run histograms AFTER balancing:")
    plot_steering_histograms(balanced_df)


if __name__ == "__main__":
    main()
