"""
LSTM Model Module for HAR (Human Activity Recognition)

This module implements an LSTM (Long Short-Term Memory) model for classifying
human activities from WiFi CSI time series data.

LSTM Overview:
A normal perceptron has no memory — each input is treated independently.
For time series data, this is a problem: to recognize walking, you need
not just the current signal value but also the values from previous timesteps.

LSTM has 2 types of memory:
  - Cell State: Long-term memory that flows through the entire sequence
  - Hidden State: Short-term memory (output of the current timestep)

And three gates that control what is remembered and forgotten:
  - Forget Gate: Which old information should be deleted?
  - Input Gate: Which new information should be stored?
  - Output Gate: What should be passed to the next timestep?
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)


class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    """
    Custom callback that prints training metrics every 10 epochs.

    Provides visibility into training progress without overwhelming output.
    """

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            total = self.params['epochs']
            print(f"Epoch {epoch + 1}/{total} - "
                  f"Train Loss: {logs['loss']:.4f} | "
                  f"Val Loss: {logs['val_loss']:.4f} | "
                  f"Train Acc: {logs['accuracy'] * 100:.1f}% | "
                  f"Val Acc: {logs['val_accuracy'] * 100:.1f}%")


class LSTMModel:
    """
    LSTM Model for HAR classification.

    This model uses a single LSTM layer to process time series data
    and capture long-term dependencies for activity recognition.
    """

    def __init__(self, input_shape=(500, 256), hidden_size=64, num_classes=5,
                 learning_rate=0.005, epochs=200, batch_size=32, patience=20):
        """
        Initialize the LSTM model with specified hyperparameters.

        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            hidden_size (int): Number of LSTM units
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for Adam optimizer
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            patience (int): Early stopping patience
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.training_time = None
        self.inference_time = None
        self.history = None

        # Build the model architecture
        self.model = Sequential([
            # LSTM layer with hidden_size units
            LSTM(hidden_size, input_shape=input_shape),

            # Output layer: softmax for probability distribution over classes
            Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train):
        """
        Train the LSTM model on the provided data.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
        """
        # Split training data for validation (20% for validation)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        # Train the model
        start = time.time()
        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[PrintEvery10Epochs(), early_stop]
        )
        self.training_time = time.time() - start

        # Display training completion information
        actual_epochs = len(self.history.history['loss'])
        if actual_epochs < self.epochs:
            print(f"\nEarly Stopping at Epoch {actual_epochs}")

        print(f"LSTM Training completed in {self.training_time:.4f} seconds")

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
        plt.savefig('lstm_training_curves.png', dpi=150)
        plt.close()

        print("Training curves saved: lstm_training_curves.png")

    def predict(self, X_test):
        """
        Generate predictions for test data.

        Args:
            X_test (np.ndarray): Test data

        Returns:
            np.ndarray: Predicted class labels
        """
        start = time.time()
        # Get probability distribution over classes
        probabilities = self.model.predict(X_test, verbose=0)
        self.inference_time = time.time() - start

        print(f"LSTM Inference completed in {self.inference_time:.4f} seconds")

        # Return the class with highest probability
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

        print(f"LSTM Accuracy: {accuracy * 100:.2f}%")

        return accuracy, predictions