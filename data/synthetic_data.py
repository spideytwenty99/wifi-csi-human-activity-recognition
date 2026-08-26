import numpy as np

# =======================
# Constants
# =======================

N_SAMPLES = 300     # Number of samples per activity
TIMESTEPS = 50      # Number of time steps per sample


"""
Synthetic CSI Data Generator

This module generates three classes of synthetic CSI-like signals:

Class 0 - No Movement
    Stable signal with only small random noise.

Class 1 - Walking
    Regular sinusoidal oscillation representing a walking pattern.

Class 2 - Waving
    Faster oscillations with larger amplitudes representing
    rapid hand movements.
"""


# =======================
# Class 0 - No Movement
# =======================

def generate_class0(n, t):
    """
    Generate samples for Class 0 (No Movement).

    The signal remains mostly stable around a constant value,
    with only small random noise added.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    t : int
        Number of time steps per sample.

    Returns
    -------
    np.ndarray
        Generated signals.
    """
    data = []

    for _ in range(n):
        noise = np.random.normal(0, 0.1, t)
        signal = 1.0 + noise

        data.append(signal)

    return np.array(data)


# =======================
# Class 1 - Walking
# =======================

def generate_class1(n, t):
    """
    Generate samples for Class 1 (Walking).

    The signal follows a regular sinusoidal pattern with
    slight variations in frequency and added noise.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    t : int
        Number of time steps per sample.

    Returns
    -------
    np.ndarray
        Generated signals.
    """
    data = []

    for _ in range(n):

        # Create equally spaced time points
        time = np.linspace(
            0,
            2 * np.pi,
            t
        )

        # Random frequency variation
        freq = np.random.uniform(
            0.8,
            1.2
        )

        noise = np.random.normal(
            0,
            0.15,
            t
        )

        signal = np.sin(
            freq * time
        ) + noise

        data.append(signal)

    return np.array(data)


# =======================
# Class 2 - Waving
# =======================

def generate_class2(n, t):
    """
    Generate samples for Class 2 (Waving).

    The signal contains faster oscillations and larger
    amplitudes than the walking class, representing
    rapid hand movements.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    t : int
        Number of time steps per sample.

    Returns
    -------
    np.ndarray
        Generated signals.
    """
    data = []

    for _ in range(n):

        # Two full periods for faster movement
        time = np.linspace(
            0,
            4 * np.pi,
            t
        )

        # Higher frequency than walking
        freq = np.random.uniform(
            2.5,
            3.5
        )

        # Larger signal amplitude
        amp = np.random.uniform(
            1.5,
            2.5
        )

        noise = np.random.normal(
            0,
            0.2,
            t
        )

        signal = (
            amp * np.sin(freq * time)
            + noise
        )

        data.append(signal)

    return np.array(data)


# ======================================
# Main Dataset Generation Function
# ======================================

def generate_data():
    """
    Generate the complete synthetic dataset.

    Creates three activity classes and combines them
    into a single dataset with corresponding labels.

    Returns
    -------
    X : np.ndarray
        Generated signal data.
    y : np.ndarray
        Class labels.
    """
    np.random.seed(42)  # Reproducibility

    X0 = generate_class0(
        N_SAMPLES,
        TIMESTEPS
    )

    X1 = generate_class1(
        N_SAMPLES,
        TIMESTEPS
    )

    X2 = generate_class2(
        N_SAMPLES,
        TIMESTEPS
    )

    # Combine all classes
    X = np.vstack([
        X0,
        X1,
        X2
    ])

    # Generate labels
    y = np.array(
        [0] * N_SAMPLES +
        [1] * N_SAMPLES +
        [2] * N_SAMPLES
    )

    return X, y


# =======================
# Test
# =======================

if __name__ == "__main__":

    X, y = generate_data()

    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    print(f"Classes: {np.unique(y)}")