import os
import pandas as pd
import matplotlib.pyplot as plt

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Select one CSV file
file_path = os.path.join(BASE_DIR, "data", "csv", "Walking_100.csv")

# Read CSV
df = pd.read_csv(file_path)

# --------------------------------------------------
# Plot one subcarrier
# --------------------------------------------------
signal = df["Subcarrier_1"]

plt.figure(figsize=(12, 5))
plt.plot(signal)

plt.title("Walking - Subcarrier 1")
plt.xlabel("Time Sample")
plt.ylabel("Amplitude")

plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot multiple subcarriers
# --------------------------------------------------
subcarriers = [
    "Subcarrier_1",
    "Subcarrier_64",
    "Subcarrier_128",
    "Subcarrier_192",
    "Subcarrier_256"
]

plt.figure(figsize=(14, 6))

for column in subcarriers:
    plt.plot(df[column], label=column)

plt.title("Walking - Multiple Subcarriers")
plt.xlabel("Time Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Heatmap
# --------------------------------------------------
plt.figure(figsize=(14, 8))

plt.imshow(
    df.T,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    interpolation="nearest"
)

plt.colorbar(label="Amplitude")

plt.title("CSI Amplitude Heatmap - Walking")
plt.xlabel("Time Sample")
plt.ylabel("Subcarrier")

plt.yticks(
    [0, 63, 127, 191, 255],
    [1, 64, 128, 192, 256]
)

plt.tight_layout()
plt.show()

# The visualization of multiple CSI subcarriers shows that different
# subcarriers exhibit different amplitude ranges and temporal variations.
# This indicates that human movement affects the WiFi channel differently
# across the frequency spectrum. Therefore, all 256 subcarriers are retained
# for subsequent preprocessing and model training.

# CSI amplitude heatmap for a walking activity. The horizontal axis represents
# time samples, while the vertical axis represents the 256 CSI subcarriers.
# Color intensity corresponds to the measured amplitude. Distinct temporal and
# spectral patterns can be observed, indicating that human movement affects
# different subcarriers differently.