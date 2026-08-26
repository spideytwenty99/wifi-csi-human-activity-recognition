"""
GRU Model Module for HAR (Human Activity Recognition)

This module implements a GRU (Gated Recurrent Unit) model for classifying
human activities from WiFi CSI time series data.

GRU (Gated Recurrent Unit) Overview:
- Like LSTM, GRU is designed for time series data
- Unlike a normal neural network, GRU can store information over multiple timesteps
- GRU has only one hidden state and uses two gates:
    - Update Gate → Decides which information to keep
    - Reset Gate → Decides which old information to forget
- Has fewer parameters than LSTM and trains faster with comparable accuracy
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    """
    Custom callback that prints training metrics every 10 epochs.

    Provides visibility into training progress without overwhelming output.
    """

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            total = self.params["epochs"]
            print(
                f"Epoch {epoch + 1}/{total} - "
                f"Train Loss: {logs['loss']:.4f} | "
                f"Val Loss: {logs['val_loss']:.4f} | "
                f"Train Acc: {logs['accuracy'] * 100:.1f}% | "
                f"Val Acc: {logs['val_accuracy'] * 100:.1f}%"
            )


class GRUModel:
    """
    GRU Model for HAR classification.

    This model uses three stacked GRU layers to process time series data
    and capture temporal dependencies for activity recognition.
    """

    def __init__(
            self,
            input_shape=(500, 256),
            hidden_size=64,
            num_classes=5,
            learning_rate=0.005,
            epochs=200,
            batch_size=32,
            patience=20,
            dropout=0.0,
    ):
        """
        Initialize the GRU model with specified hyperparameters.

        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            hidden_size (int): Number of GRU units in the second and third layers
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for Adam optimizer
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            patience (int): Early stopping patience
            dropout (float): Dropout rate for regularization
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout

        self.training_time = None
        self.inference_time = None
        self.history = None

        # Build the model architecture with three GRU layers
        self.model = Sequential([
            Input(shape=input_shape),

            # First GRU layer with 128 units
            # return_sequences=True to pass sequence to next GRU layer
            GRU(128, dropout=self.dropout, return_sequences=True),

            # Second GRU layer with hidden_size units
            GRU(hidden_size, dropout=self.dropout, return_sequences=True),

            # Third GRU layer with hidden_size units
            GRU(hidden_size, dropout=self.dropout),

            # Output layer
            Dense(num_classes, activation="softmax")
        ])

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Alternative architecture (commented for reference)
        # self.model = Sequential([
        #     Input(shape=input_shape),
        #     GRU(128, dropout=self.dropout, return_sequences=True),
        #     GRU(64),
        #     Dense(num_classes, activation="softmax")
        # ])
        #
        # self.model.compile(
        #     optimizer=Adam(learning_rate=learning_rate),
        #     loss="sparse_categorical_crossentropy",
        #     metrics=["accuracy"]
        # )

    def train(self, X_train, y_train):
        """
        Train the GRU model on the provided data.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
        """
        # Split training data for validation (20% for validation with stratification)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

        # Train the model
        start = time.time()
        self.history = self.model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[
                PrintEvery10Epochs(),
                early_stop
            ]
        )
        self.training_time = time.time() - start

        # Display training completion information
        actual_epochs = len(self.history.history["loss"])
        if actual_epochs < self.epochs:
            print(f"\nEarly Stopping at Epoch {actual_epochs}")

        print(f"GRU Training completed in {self.training_time:.4f} seconds")

    def plot_history(self):
        """
        Plot and save training history curves.

        Creates two side-by-side plots showing loss and accuracy
        for both training and validation sets.
        """
        h = self.history.history
        epochs = range(1, len(h["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(epochs, h["loss"], label="Train Loss", linewidth=2)
        ax1.plot(epochs, h["val_loss"], label="Validation Loss", linewidth=2)
        ax1.set_title("Train Loss vs Validation Loss", fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, h["accuracy"], label="Train Accuracy", linewidth=2)
        ax2.plot(epochs, h["val_accuracy"], label="Validation Accuracy", linewidth=2)
        ax2.set_title("Train Accuracy vs Validation Accuracy", fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("gru_training_curves.png", dpi=150)
        plt.close()

        print("Training curves saved as gru_training_curves.png")

    def predict(self, X_test):
        """
        Generate predictions for test data.

        Args:
            X_test (np.ndarray): Test data

        Returns:
            np.ndarray: Predicted class labels
        """
        start = time.time()
        probabilities = self.model.predict(X_test, verbose=0)
        self.inference_time = time.time() - start

        print(f"GRU Prediction completed in {self.inference_time:.4f} seconds")

        return np.argmax(probabilities, axis=1)

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

        print(f"GRU Accuracy: {accuracy * 100:.2f}%")

        return accuracy, predictions