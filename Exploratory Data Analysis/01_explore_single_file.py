import pandas as pd

# Path to one specific CSV file
file_path = "../data/csv/Walking_45.csv"

# Read the CSV file
df = pd.read_csv(file_path)

print("DATASET INFORMATION")

print("Shape:", df.shape)
rows, cols = df.shape

print("Number of rows:", rows)
print("Number of columns:", cols)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Overall statistics
print("\nOverall Minimum:", df.min().min())
print("Overall Maximum:", df.max().max())
print("Overall Mean:", df.mean().mean())
print("Overall Standard Deviation:", df.std().mean())