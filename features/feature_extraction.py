"""
Feature Extraction Module for HAR (Human Activity Recognition)

This module extracts statistical features from CSI time series data
to prepare it for machine learning models like SVM and k-NN.

After preprocessing, data has shape (n_samples, 50) - raw time steps per sample.
SVM and k-NN cannot handle raw time series data well. They require
meaningful numeric values per sample - called features.

For each signal, five statistical features are extracted:
    - Mean: Average signal value
    - Variance: How much the signal fluctuates
    - Maximum: Largest amplitude
    - Minimum: Smallest amplitude
    - Energy: Sum of squared values (total signal strength)

Each sample with 50 timesteps becomes a vector of 5 numbers per subcarrier.
This is called a feature vector.
"""

import numpy as np


def extract_features(X):
    """
    Extract statistical features from each time series sample.

    For each of the 256 subcarriers, five features are computed:
    mean, variance, maximum, minimum, and energy.

    Args:
        X (np.ndarray): Input data with shape (n_samples, 500, 256)
                       where 500 is timesteps and 256 is subcarriers

    Returns:
        np.ndarray: Feature matrix with shape (n_samples, 1280)
                   where 1280 = 256 subcarriers * 5 features per subcarrier
    """
    features = []

    # Iterate through each sample
    for sample in X:
        sample_features = []

        # Iterate through all 256 subcarriers
        for i in range(256):
            subcarrier = sample[:, i]

            # Calculate statistical features
            mean_value = np.mean(subcarrier)
            variance = np.var(subcarrier)
            maximum = np.max(subcarrier)
            minimum = np.min(subcarrier)
            energy = np.sum(subcarrier ** 2)

            # Append all features to the sample's feature vector
            sample_features.extend([mean_value, variance, maximum, minimum, energy])

        features.append(sample_features)

    return np.array(features)


# Test the feature extraction function
if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.synthetic_data import generate_data

    # Generate synthetic data
    X, y = generate_data()

    # Extract features
    F = extract_features(X)

    # Display results
    print(f"X Shape before: {X.shape}")
    print(f"F Shape after: {F.shape}")
    print(f"\nExample Feature Vector (Sample 0): {F[0]}")
    print(f"Meaning: [Mean, Variance, Max, Min, Energy] repeated for all 256 subcarriers")