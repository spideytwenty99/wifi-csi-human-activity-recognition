"""
CSI Signal Visualization Script

This script visualizes WiFi CSI (Channel State Information) amplitude data
from a single CSV file, including single subcarrier plots, multiple subcarrier
overlays, and a full heatmap of all 256 subcarriers.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Select one CSV file for visualization
file_path = os.path.join(BASE_DIR, "data", "csv", "Walking_100.csv")

# Read CSV file
df = pd.read_csv(file_path)


# Plot a single subcarrier to examine its temporal pattern
signal = df["Subcarrier_1"]

plt.figure(figsize=(12, 5))
plt.plot(signal, linewidth=2)

plt.title("Walking - Subcarrier 1", fontweight='bold')
plt.xlabel("Time Sample")
plt.ylabel("Amplitude")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Plot multiple subcarriers to compare their patterns
subcarriers = [
    "Subcarrier_1",
    "Subcarrier_64",
    "Subcarrier_128",
    "Subcarrier_192",
    "Subcarrier_256"
]

plt.figure(figsize=(14, 6))

for column in subcarriers:
    plt.plot(df[column], label=column, linewidth=2)

plt.title("Walking - Multiple Subcarriers", fontweight='bold')
plt.xlabel("Time Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Plot heatmap showing all 256 subcarriers over time
plt.figure(figsize=(14, 8))

plt.imshow(
    df.T,                          # Transpose: subcarriers as rows, time as columns
    aspect="auto",                 # Automatically adjust aspect ratio
    origin="lower",                # Start from bottom
    cmap="viridis",                # Color map for amplitude values
    interpolation="nearest"        # No interpolation between pixels
)

plt.colorbar(label="Amplitude")

plt.title("CSI Amplitude Heatmap - Walking", fontweight='bold')
plt.xlabel("Time Sample")
plt.ylabel("Subcarrier")

# Set y-axis ticks to show selected subcarrier indices
plt.yticks(
    [0, 63, 127, 191, 255],
    [1, 64, 128, 192, 256]
)

plt.tight_layout()
plt.show()

