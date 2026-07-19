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

# ======================================================
# SIGNAL COMPARISON
# ======================================================

plt.figure(figsize=(12, 12))

plot_number = 1

for activity, filename in files.items():

    plt.subplot(5, 1, plot_number)

    file_path = os.path.join(dataset_path, filename)

    df = pd.read_csv(file_path)

    signal = df["Subcarrier_64"]

    plt.plot(signal)

    plt.title(activity)
    plt.xlabel("Time Sample")
    plt.ylabel("Amplitude")

    plt.ylim(0, 1500)

    plt.grid(True)

    plot_number += 1

plt.tight_layout()
plt.show()

# ======================================================
# BOXPLOT
# ======================================================

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

plt.title("Comparison of Subcarrier_64")

plt.xlabel("Activity")
plt.ylabel("Amplitude")

plt.xticks(
    range(1, len(activity_names) + 1),
    activity_names
)

plt.grid(True)

plt.tight_layout()
plt.show()

# ======================================================
# HISTOGRAMS
# ======================================================

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
        edgecolor="black"
    )

    plt.title(activity)
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    plt.xlim(0, 1500)

    plot_number += 1

plt.tight_layout()
plt.show()

# The boxplot of Subcarrier 64 shows noticeable differences in amplitude
# distributions across activities. Walking exhibits the highest median
# amplitude and the largest variability, while Sitting and Standing have
# comparatively lower medians and similar distributions.

# Histograms of Subcarrier 64 reveal positively skewed amplitude distributions.
# These observations support the application of feature scaling before
# training machine learning models. Standardization (StandardScaler) was
# therefore applied for the SVM.