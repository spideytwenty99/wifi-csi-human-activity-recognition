"""
WiFi-CSI Dataset Report Script

This script analyzes the Wi-Fi CSI (Channel State Information) dataset,
providing statistics about the data files for each activity class.
"""

import pandas as pd
import os


# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
dataset_path = os.path.join(BASE_DIR, "data", "csv")

# List of activity classes in the dataset
activities = [
    "Empty",
    "Lying",
    "Sitting",
    "Standing",
    "Walking",
]


print("Overall WiFi-CSI Dataset Report")

# Get all CSV files from the dataset directory
all_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

total_files = len(all_files)


# Analyze each activity separately
for activity in activities:
    print(f"\nActivity: {activity}")

    # Filter files belonging to this activity
    csv_files = [f for f in all_files if f.startswith(activity + "_")]

    print("Number of CSV files:", len(csv_files))

    # Lists to store rows and columns counts for each file
    rows = []
    cols = []

    missing_values = 0
    duplicate_rows = 0
    corrupted_files = []

    # Read and analyze each CSV file
    for file in csv_files:
        file_path = os.path.join(dataset_path, file)

        try:
            df = pd.read_csv(file_path)

            rows.append(df.shape[0])
            cols.append(df.shape[1])

            missing_values += df.isnull().sum().sum()
            duplicate_rows += df.duplicated().sum()

        except Exception:
            corrupted_files.append(file)

    # Display statistics for this activity
    if len(rows) > 0:
        print("Average Rows      :", sum(rows) / len(rows))
        print("Minimum Rows      :", min(rows))
        print("Maximum Rows      :", max(rows))
        print("Average Columns   :", sum(cols) / len(cols))

    print("Missing Values    :", missing_values)
    print("Duplicate Rows    :", duplicate_rows)
    print("Corrupted Files   :", len(corrupted_files))

    # Display list of corrupted files if any
    if corrupted_files:
        print("\nCorrupted Files:")
        for f in corrupted_files:
            print(f"  {f}")

# Display total file count
print(f"\nTotal CSV files: {total_files}")


"""
Dataset Description

The dataset consists of WiFi CSI amplitude recordings for five human activities:
- Empty
- Lying
- Sitting
- Standing
- Walking

Each recording is stored as a CSV file containing:
- 500 time samples
- 256 CSI amplitude subcarriers

Dataset Integrity Check Results:
- All files have a consistent shape
- No missing values found
- No duplicate rows found
- No corrupted files found
"""