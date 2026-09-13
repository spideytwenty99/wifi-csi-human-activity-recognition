"""
GRU Hyperparameter Tuning for HAR (Human Activity Recognition)

This script performs a grid search over hyperparameters for a GRU model
to find the optimal configuration for activity recognition from sensor data.
"""

import itertools
import gc
import pandas as pd
import tensorflow as tf
from numpy import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_real_data import load_real_data
from preprocessing.preprocess import preprocess
from models.gru_model import GRUModel


# Set random seed for reproducibility across all libraries
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 60)
print("          GRU Hyperparameter Tuning")
print("=" * 60)


# Load and preprocess the real dataset
X, y, encoder = load_real_data()
X_train, X_test, y_train, y_test, scaler = preprocess(X, y)


# Define hyperparameter search space
hidden_sizes = [64]              # Number of GRU units
learning_rates = [0.005]         # Learning rate for Adam optimizer
batch_sizes = [16, 32, 64]       # Batch size for training
dropouts = [0.0, 0.1, 0.2]       # Dropout rate for regularization
patiences = [20, 30]             # Early stopping patience


# Initialize list to store results from each experiment
results = []

# Calculate total number of experiments
total = (
    len(hidden_sizes)
    * len(learning_rates)
    * len(batch_sizes)
    * len(dropouts)
    * len(patiences)
)

experiment = 1


# Iterate over all hyperparameter combinations
for hidden_size, lr, batch_size, dropout, patience in itertools.product(
        hidden_sizes,
        learning_rates,
        batch_sizes,
        dropouts,
        patiences
):

    # Display current experiment information
    print("\n" + "=" * 60)
    print(f"Experiment {experiment}/{total}")
    print(f"Hidden Size   : {hidden_size}")
    print(f"Learning Rate : {lr}")
    print(f"Batch Size    : {batch_size}")
    print(f"Dropout       : {dropout}")
    print(f"Patience      : {patience}")

    # Create and train GRU model with current hyperparameters
    model = GRUModel(
        hidden_size=hidden_size,
        learning_rate=lr,
        batch_size=batch_size,
        epochs=200,
        patience=patience,
        dropout=dropout
    )

    model.train(X_train, y_train)

    # Evaluate model on test set
    accuracy, _ = model.evaluate(X_test, y_test)

    # Store results for this experiment
    results.append({
        "Hidden Size": hidden_size,
        "Learning Rate": lr,
        "Batch Size": batch_size,
        "Dropout": dropout,
        "Patience": patience,
        "Accuracy": accuracy,
        "Training Time (s)": round(model.training_time, 2),
        "Inference Time (s)": round(model.inference_time, 2)
    })

    # Free memory before next experiment to prevent OOM errors
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    experiment += 1


# Convert results to DataFrame for easier analysis
results = pd.DataFrame(results)

# Sort by accuracy (highest first)
results = results.sort_values(
    by="Accuracy",
    ascending=False
)


# Display top configurations
print("\n")
print("=" * 60)
print("Top Configurations")
print("=" * 60)

print(results)

# Save results to CSV file for later reference
results.to_csv("gru_hyperparameter_results.csv", index=False)

print("\nResults saved to gru_hyperparameter_results.csv")