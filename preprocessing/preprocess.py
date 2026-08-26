import numpy as np
from sklearn.model_selection import train_test_split  # Split data into training and test sets
from sklearn.preprocessing import StandardScaler      # Normalize data

"""
Data preprocessing.

This module performs two main tasks:

1. Normalization
   All feature values are scaled to a common range.
   Machine learning models generally perform better when
   features have similar magnitudes. Without normalization,
   features with larger values may dominate the learning process.

2. Train/Test Split
   The dataset is divided into:
   - 80% training data
   - 20% testing data

   The model learns from the training data and is evaluated
   on the test data, which it has never seen before.
   This helps determine whether the model has learned
   meaningful patterns or simply memorized the training data.
"""


def preprocess(X, y, test_size=0.2):
    """
    Perform train/test split and feature normalization.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Class labels.
    test_size : float, optional
        Fraction of the dataset used for testing (default: 0.2).

    Returns
    -------
    X_train : np.ndarray
        Normalized training data.
    X_test : np.ndarray
        Normalized test data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    scaler : StandardScaler
        Fitted scaler used for normalization.
    """

    # 20% of the data is used for testing, 80% for training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,  # Ensures reproducibility
        stratify=y        # Preserves class distribution in both sets
    )

    # Reshape data to 2D for StandardScaler
    X_train_flat = X_train.reshape(-1, 256)
    X_test_flat = X_test.reshape(-1, 256)

    # Normalize features
    scaler = StandardScaler()

    # Compute mean and standard deviation from training data
    X_train = scaler.fit_transform(X_train_flat)

    # Apply the same transformation to test data
    X_test = scaler.transform(X_test_flat)

    # Reshape data back to 3D
    n_train = X_train_flat.shape[0] // 500
    n_test = X_test_flat.shape[0] // 500

    X_train = X_train.reshape(n_train, 500, 256)
    X_test = X_test.reshape(n_test, 500, 256)

    return X_train, X_test, y_train, y_test, scaler



# Test
if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from data.synthetic_data import generate_data

    X, y = generate_data()

    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"y_train Classes: {np.unique(y_train, return_counts=True)}")
    print(f"y_test  Classes: {np.unique(y_test, return_counts=True)}")