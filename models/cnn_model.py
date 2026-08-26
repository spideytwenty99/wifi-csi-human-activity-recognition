"""
CNN Model Module for HAR (Human Activity Recognition)

This module implements a 1D Convolutional Neural Network for classifying
human activities from Wi-Fi CSI time series data.

Architecture Overview:
- Conv1D: Filters slide over timesteps to detect local patterns
- MaxPooling: Reduces feature map size, keeps strongest values per window
- Flatten: Converts 2D feature maps to 1D vector
- Dense: Fully connected layer for final classification
- Dropout: Randomly deactivates 30% of neurons to prevent overfitting
- Softmax: Outputs probability distribution over classes

How CNN Works:
A 1D CNN uses filters (kernels) that slide over the input to search for
local patterns. For time series data, a filter looks at 3 or 5 consecutive
timesteps to find characteristic patterns - such as the rhythmic oscillation
of walking.
"""

import os
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

SEED = 42

# Set seeds for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


class CNNModel:
    """
    1D Convolutional Neural Network for HAR classification.

    This model processes CSI time series data using two convolutional
    blocks followed by fully connected layers for classification.
    """

    def __init__(self, input_shape=(500, 256), num_classes=5,
                 filters=128, kernel_size=3, learning_rate=0.0005,
                 epochs=50, batch_size=16):
        """
        Initialize the CNN model with specified hyperparameters.

        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            num_classes (int): Number of output classes
            filters (int): Number of filters in convolutional layers
            kernel_size (int): Size of the sliding window in Conv1D
            learning_rate (float): Learning rate for Adam optimizer
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.training_time = None
        self.inference_time = None
        self.history = None

        # Build the model architecture
        self.model = Sequential([
            # First convolutional block
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                input_shape=input_shape
            ),
            MaxPooling1D(pool_size=2),  # Halves the time dimension

            # Second convolutional block
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu'
            ),
            MaxPooling1D(pool_size=2),  # Halves the time dimension again

            # Fully connected layers
            Flatten(),  # Converts 2D to 1D
            Dense(128, activation='relu'),
            Dropout(0.3),  # 30% dropout for regularization
            Dense(num_classes, activation='softmax')  # Output layer
        ])

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train):
        """
        Train the CNN model on the provided data.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
        """
        # Split training data for validation (20% for validation)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.20, random_state=SEED
        )

        # Train the model
        start = time.time()
        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1  # Prints progress for each epoch
        )
        self.training_time = time.time() - start

        print(f"CNN Training completed in {self.training_time:.4f} seconds")

    def plot_history(self):
        """
        Plot and save training history curves.

        Creates two side-by-side plots showing loss and accuracy
        for both training and validation sets.
        """
        h = self.history.history
        epochs = range(1, len(h['loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(epochs, h['loss'], label='Train Loss', linewidth=2)
        ax1.plot(epochs, h['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title('Train Loss vs Val Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, h['accuracy'], label='Train Acc', linewidth=2)
        ax2.plot(epochs, h['val_accuracy'], label='Val Acc', linewidth=2)
        ax2.set_title('Train Accuracy vs Val Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('cnn_training_curves.png', dpi=150)
        plt.close()

        print("Training curves saved: cnn_training_curves.png")

    def predict(self, X_test):
        """
        Generate predictions for test data.

        Args:
            X_test (np.ndarray): Test data

        Returns:
            np.ndarray: Predicted class labels
        """
        start = time.time()
        raw_predictions = self.model.predict(X_test, verbose=0)
        self.inference_time = time.time() - start

        print(f"CNN Inference completed in {self.inference_time:.4f} seconds")

        return np.argmax(raw_predictions, axis=1)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.

        Args:
            X_test (np.ndarray): Test data
            y_test (np.ndarray): True labels

        Returns:
            tuple: (accuracy, predictions)
                - accuracy (float): Test accuracy as a fraction
                - predictions (np.ndarray): Predicted class labels
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"CNN Accuracy: {accuracy * 100:.2f}%")

        return accuracy, predictions