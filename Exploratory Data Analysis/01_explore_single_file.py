"""
CSV Data Inspection Script

This script loads and analyzes a single CSV file from the HAR dataset,
displaying  information about its structure and content.
"""

import pandas as pd


# Path to one specific CSV file
file_path = "../data/csv/Walking_45.csv"

# Read the CSV file
df = pd.read_csv(file_path)


print("DATASET INFORMATION")

# Display basic shape information
print("Shape:", df.shape)
rows, cols = df.shape
print("Number of rows:", rows)
print("Number of columns:", cols)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display data types of each column
print("\nData Types:")
print(df.dtypes)

# Display first and last few rows to understand data structure
print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Display summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Display overall statistics across all columns
print("\nOverall Minimum:", df.min().min())
print("Overall Maximum:", df.max().max())
print("Overall Mean:", df.mean().mean())
print("Overall Standard Deviation:", df.std().mean())