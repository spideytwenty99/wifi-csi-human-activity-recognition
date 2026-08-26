import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Path to CSV files
DATA_PATH = "data/csv"


def load_real_data():
    """
 Load and preprocess sensor data from CSV files.

    Reads all CSV files from the configured data directory, extracts
    class labels from filenames, and encodes them as integers.

    Returns:
        tuple: (X, y, encoder)
            - X (np.ndarray): Array of sensor data with shape (n_samples, n_timesteps, n_features)
            - y (np.ndarray): Encoded class labels as integers
            - encoder (LabelEncoder): Fitted label encoder for decoding predictions
    """


    # Lists for signal data and corresponding labels
    X = []
    y = []

    # Iterate through CSV files in the data directory
    for file in sorted(os.listdir(DATA_PATH)):
        if file.endswith(".csv"):

            # Extract class label from filename
            label = file.split("_")[0]

            # Read CSV file
            filepath = os.path.join(DATA_PATH, file)
            df = pd.read_csv(filepath)

            # Store data and label
            X.append(df)
            y.append(label)

            # Optional debugging output
            # print(f"{file} → Label: {label}, Shape: {df.shape}")

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Dataset summary
    print("\n========== Dataset Summary ==========")
    print(f"Total Files Loaded : {len(X)}")
    print(f"Each File Shape    : {X[0].shape}")
    print(f"Dataset Shape      : {X.shape}")
    print(f"Classes            : {np.unique(y)}")
    print("=====================================")

    # Encode string labels into numerical labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print(f"Labels als Zahlen: {np.unique(y)}")
    print(f"Bedeutung: {encoder.classes_}")

    # Return data, encoded labels, and encoder
    return X, y, encoder