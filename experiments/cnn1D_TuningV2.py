# IMPORT LIBRARIES

import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    ParameterGrid
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)


# REPRODUCIBILITY
"""
Set random seeds to ensure reproducible results.

Using fixed seeds helps guarantee that:
- Dataset splitting remains consistent
- Weight initialization is repeatable
- Hyperparameter experiments are comparable
"""

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# LOAD AND PREPROCESS DATA

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_real_data import load_real_data
from preprocessing.preprocess import preprocess

print("Loading dataset...")

# Load all CSV files and labels
X, y, encoder = load_real_data()

# Apply train/test split and normalization
X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

# Create validation set from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Training Samples   : {len(X_train)}")
print(f"Validation Samples : {len(X_val)}")
print(f"Test Samples       : {len(X_test)}")

# Dataset dimensions
timesteps = X_train.shape[1]
features = X_train.shape[2]
n_outputs = len(np.unique(y_train))

print("\nDataset ready for CNN training.")
print("Training Shape   :", X_train.shape)
print("Validation Shape :", X_val.shape)
print("Test Shape       :", X_test.shape)


# HYPERPARAMETER SEARCH SPACE

"""
All combinations generated from this parameter grid
will be evaluated during hyperparameter tuning.
"""

param_grid = {
    "learning_rate": [0.01, 0.001, 0.0005, 0.0001],
    "batch_size": [16, 32],
    "filters": [64, 128],
    "kernel_size": [3, 5],
    "dropout": [0.2, 0.3]
}

grid = ParameterGrid(param_grid)

print(f"\nTotal Experiments: {len(grid)}")

# Store experiment results
results = []


# EARLY STOPPING

"""
Stop training when validation loss no longer improves.

Benefits:
- Prevents overfitting
- Reduces unnecessary training time
- Restores the best model weights automatically
"""

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# HYPERPARAMETER TUNING

for i, params in enumerate(grid):

    # Reset seeds before every experiment
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("\n=================================================")
    print(f"Experiment {i + 1}/{len(grid)}")
    print("Current Parameters")
    print(params)

    # -------------------------------------------------
    # Build CNN Model
    # -------------------------------------------------

    model = Sequential([
        Conv1D(
            filters=params["filters"],
            kernel_size=params["kernel_size"],
            activation="relu",
            input_shape=(timesteps, features)
        ),

        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=params["filters"] * 2,
            kernel_size=params["kernel_size"],
            activation="relu"
        ),

        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            params["dropout"]
        ),

        Dense(
            n_outputs,
            activation="softmax"
        )
    ])

    # Compile Model
    optimizer = Adam(
        learning_rate=params["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    # Train Model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0
    )


    # Evaluate Model
    train_loss, train_accuracy = model.evaluate(
        X_train,
        y_train,
        verbose=0
    )

    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )


    # Store Results
    results.append({
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "filters": params["filters"],
        "kernel_size": params["kernel_size"],
        "dropout": params["dropout"],
        "epochs_trained": len(history.history["loss"]),
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "test_loss": test_loss
    })

    print(f"Training Accuracy   : {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")


# SORT RESULTS BY VALIDATION ACCURACY
results = sorted(
    results,
    key=lambda x: x["validation_accuracy"],
    reverse=True
)


# TOP HYPERPARAMETER COMBINATIONS
print("\n")
print("=" * 70)
print("TOP HYPERPARAMETER COMBINATIONS")
print("=" * 70)

for i, result in enumerate(results[:10]):

    print(f"\nRank {i + 1}")
    print(f"Learning Rate       : {result['learning_rate']}")
    print(f"Batch Size          : {result['batch_size']}")
    print(f"Filters             : {result['filters']}")
    print(f"Kernel Size         : {result['kernel_size']}")
    print(f"Dropout             : {result['dropout']}")
    print(f"Training Accuracy   : {result['train_accuracy'] * 100:.2f}%")
    print(f"Validation Accuracy : {result['validation_accuracy'] * 100:.2f}%")
    print(f"Test Accuracy       : {result['test_accuracy'] * 100:.2f}%")


# BEST HYPERPARAMETER COMBINATION
best_params = results[0]

print("\n")
print("=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)

print(f"Learning Rate       : {best_params['learning_rate']}")
print(f"Batch Size          : {best_params['batch_size']}")
print(f"Filters             : {best_params['filters']}")
print(f"Kernel Size         : {best_params['kernel_size']}")
print(f"Dropout             : {best_params['dropout']}")
print(f"Training Accuracy   : {best_params['train_accuracy'] * 100:.2f}%")
print(f"Validation Accuracy : {best_params['validation_accuracy'] * 100:.2f}%")
print(f"Test Accuracy       : {best_params['test_accuracy'] * 100:.2f}%")