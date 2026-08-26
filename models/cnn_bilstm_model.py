"""
CNN + BiLSTM Model Module for HAR (Human Activity Recognition)

This module implements a hybrid CNN-BiLSTM architecture for classifying
human activities from WiFi CSI time series data.

Why BiLSTM?
- A standard LSTM processes sequences only forward (past → future)
- A BiLSTM adds a second LSTM layer that runs backward (future → past)
- Both outputs are merged → the model knows both past and future context
  for each timestep

Example: The end of a "sitting" movement provides context about the beginning.
Elkelany et al. (2023) show that BiLSTM outperforms unidirectional LSTM for
WiFi-CSI HAR because activities have temporally symmetric patterns.

Difference from CNN+LSTM (Opt 9):
- LSTM(64)              → Output: (64,)
- Bidirectional(LSTM(64)) → Output: (128,)  [64 forward + 64 backward]

The Dense layer automatically receives double the input size — no further
adjustment needed. CNN layers and all regularization measures remain
identical to Opt 9.

Architecture Overview:
  Input (500, 256)
  → Conv1D(128) + BatchNorm + MaxPool  → (249, 128)   coarse patterns
  → Dropout(0.3)
  → Conv1D(128) + BatchNorm + MaxPool  → (123, 128)   complex patterns
  → Dropout(0.3)
  → Conv1D(128) + BatchNorm + MaxPool  → (60, 128)    abstract features
  → Dropout(0.3)
  → Bidirectional(LSTM(64))            → (128,)       forward + backward
  → Dropout(0.3)
  → Dense(5, softmax)                  → prediction
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)


class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    """
    Custom callback that prints training metrics every 10 epochs.

    Provides visibility into training progress without overwhelming output.
    """

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            total = self.params['epochs']
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            print(f"Epoch {epoch + 1}/{total} - "
                  f"Train Loss: {logs['loss']:.4f} | "
                  f"Val Loss: {logs['val_loss']:.4f} | "
                  f"Train Acc: {logs['accuracy'] * 100:.1f}% | "
                  f"Val Acc: {logs['val_accuracy'] * 100:.1f}% | "
                  f"LR: {lr:.6f}")


class CNNBiLSTMModel:
    """
    Hybrid CNN-BiLSTM model for HAR classification.

    This model combines convolutional layers for spatial feature extraction
    with bidirectional LSTM layers for capturing temporal dependencies in
    both forward and backward directions.
    """

    def __init__(self, input_shape=(500, 256), num_classes=5,
                 filters=128, kernel_size=3, hidden_size=64,
                 dropout_rate=0.3,
                 learning_rate=0.0005, epochs=100, batch_size=32, patience=20):
        """
        Initialize the CNN-BiLSTM model with specified hyperparameters.

        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            num_classes (int): Number of output classes
            filters (int): Number of filters in convolutional layers
            kernel_size (int): Size of the sliding window in Conv1D
            hidden_size (int): Number of LSTM units (forward and backward)
            dropout_rate (float): Dropout rate for regularization
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
            # CNN Block 1: Coarse features
            Conv1D(filters=filters, kernel_size=kernel_size,
                   activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),  # 500 → 249
            Dropout(dropout_rate),

            # CNN Block 2: Complex patterns
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),  # 249 → 123
            Dropout(dropout_rate),

            # CNN Block 3: Abstract high-level features
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),  # 123 → 60
            Dropout(dropout_rate),

            # BiLSTM: forward + backward processing
            # Bidirectional() wraps the LSTM and adds a second direction
            # Output: 64 (forward) + 64 (backward) = 128 values per sample
            # Provides more context than unidirectional LSTM
            Bidirectional(LSTM(hidden_size)),
            Dropout(dropout_rate),

            # Output layer
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
        Train the CNN-BiLSTM model on the provided data.

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

        # ReduceLROnPlateau: halves learning rate when val_loss stagnates
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )

        # Train the model
        start = time.time()
        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[PrintEvery10Epochs(), early_stop, reduce_lr]
        )
        self.training_time = time.time() - start

        # Display training completion information
        actual_epochs = len(self.history.history['loss'])
        if actual_epochs < self.epochs:
            print(f"\nEarly Stopping at Epoch {actual_epochs}")

        print(f"CNN+BiLSTM Training completed in {self.training_time:.4f} seconds")

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
        plt.savefig('cnn_bilstm_training_curves.png', dpi=150)
        plt.close()

        print("Training curves saved: cnn_bilstm_training_curves.png")

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

        print(f"CNN+BiLSTM Inference completed in {self.inference_time:.4f} seconds")

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

        print(f"CNN+BiLSTM Accuracy: {accuracy * 100:.2f}%")

        return accuracy, predictions