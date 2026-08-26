"""
CSI Signal Comparison and Analysis Script

This script visualizes and compares Wi-Fi CSI amplitude data across all five
activity classes using signal plots, boxplots, and histograms to understand
the differences in signal characteristics between activities.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset folder
dataset_path = os.path.join(BASE_DIR, "data", "csv")

# One representative file for each activity
files = {
    "Empty": "Empty_Amp_10.csv",
    "Lying": "Lying_Amp_10.csv",
    "Sitting": "Sitting_Amp_10.csv",
    "Standing": "Standing_Amp_10.csv",
    "Walking": "Walking_10.csv"
}


# Plot signal comparison for all activities (Subcarrier_64)
plt.figure(figsize=(12, 12))

plot_number = 1

for activity, filename in files.items():
    plt.subplot(5, 1, plot_number)

    file_path = os.path.join(dataset_path, filename)
    df = pd.read_csv(file_path)

    signal = df["Subcarrier_64"]
    plt.plot(signal, linewidth=2)

    plt.title(activity, fontweight='bold')
    plt.xlabel("Time Sample")
    plt.ylabel("Amplitude")

    plt.ylim(0, 1500)
    plt.grid(True, alpha=0.3)

    plot_number += 1

plt.tight_layout()
plt.show()


# Plot boxplot to compare amplitude distributions across activities
boxplot_data = []
activity_names = []

for activity, filename in files.items():
    file_path = os.path.join(dataset_path, filename)
    df = pd.read_csv(file_path)

    signal = df["Subcarrier_64"]
    boxplot_data.append(signal)
    activity_names.append(activity)

plt.figure(figsize=(10, 6))
plt.boxplot(boxplot_data)

plt.title("Comparison of Subcarrier_64 Across Activities", fontweight='bold')
plt.xlabel("Activity")
plt.ylabel("Amplitude")

plt.xticks(
    range(1, len(activity_names) + 1),
    activity_names
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Plot histograms to examine amplitude distributions per activity
plt.figure(figsize=(12, 8))

plot_number = 1

for activity, filename in files.items():
    plt.subplot(3, 2, plot_number)

    file_path = os.path.join(dataset_path, filename)
    df = pd.read_csv(file_path)

    signal = df["Subcarrier_64"]

    plt.hist(
        signal,
        bins=30,
        edgecolor="black",
        alpha=0.7
    )

    plt.title(activity, fontweight='bold')
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    plt.xlim(0, 1500)
    plt.grid(True, alpha=0.3)

    plot_number += 1

plt.tight_layout()
plt.show()

