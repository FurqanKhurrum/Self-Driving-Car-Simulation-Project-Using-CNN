# scripts/explore_data.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- CONFIG ----
CSV_PATH = os.path.join("data", "driving_log.csv")

# Anything with |steering| < ZERO_THRESH is considered "straight"
ZERO_THRESH = 0.02

# Keep only this fraction of straight samples (e.g., 0.2 = keep 20%)
KEEP_PROB_STRAIGHT = 0.2


def main():
    # Udacity format: center, left, right, steering, throttle, brake, speed
    df = pd.read_csv(CSV_PATH, header=None)
    df.columns = ["center", "left", "right",
                  "steering", "throttle", "brake", "speed"]

    print("=== Original data ===")
    print(df.head())
    print("Total samples:", len(df))

    # ------------ ORIGINAL HISTOGRAM ------------
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.hist(df["steering"], bins=31)
    plt.title("Steering Angle Distribution (Original)")
    plt.xlabel("Steering")
    plt.ylabel("Count")

    # ------------ DOWNSAMPLING LOGIC ------------
    # Identify "straight" vs "turn" rows
    straight_mask = df["steering"].abs() < ZERO_THRESH
    df_straight = df[straight_mask]
    df_turns = df[~straight_mask]

    print("\nStraight samples (|angle| < {}): {}".format(
        ZERO_THRESH, len(df_straight)))
    print("Turning samples (other):", len(df_turns))

    # Randomly keep only a fraction of straight samples
    df_straight_down = df_straight.sample(
        frac=KEEP_PROB_STRAIGHT,
        random_state=42
    )

    # Combine downsampled straights + all turns
    df_balanced = pd.concat([df_turns, df_straight_down], axis=0)
    df_balanced = df_balanced.sample(frac=1.0, random_state=42)  # shuffle

    print("\n=== After downsampling straights ===")
    print("Straight kept:", len(df_straight_down))
    print("Turns:", len(df_turns))
    print("Total balanced samples:", len(df_balanced))

    # ------------ BALANCED HISTOGRAM ------------
    plt.subplot(2, 1, 2)
    plt.hist(df_balanced["steering"], bins=31)
    plt.title(
        f"Steering Angle Distribution (Balanced, keep {KEEP_PROB_STRAIGHT*100:.0f}% straights)"
    )
    plt.xlabel("Steering")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
