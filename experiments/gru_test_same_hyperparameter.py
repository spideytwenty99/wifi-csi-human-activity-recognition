"""
GRU (Gated Recurrent Unit)

A GRU is a simplified version of an LSTM network.

Like LSTMs, GRUs are designed to process sequential data
and retain information across multiple time steps.

Unlike an LSTM, a GRU does not maintain a separate cell state.
Instead, it uses a single hidden state together with two gates:

    Update Gate
        Controls how much information from previous
        time steps should be retained.

    Reset Gate
        Controls how much past information should be ignored.

Because GRUs contain fewer parameters than LSTMs,
they often train faster while achieving comparable
classification performance.

Note:
    This implementation uses the same hyperparameters
    as the LSTM model to enable a fair comparison.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_real_data import load_real_data
from preprocessing.preprocess import preprocess


# Set random seed for reproducibility
tf.random.set_seed(42)


class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    """
    Display training progress every 10 epochs.
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
    GRU-based classifier for CSI activity recognition.
    """

    def __init__(
        self,
        input_shape=(500, 256),
        hidden_size=64,
        num_classes=5,
        learning_rate=0.005,
        epochs=200,
        batch_size=32,
        patience=20
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.training_time = None
        self.inference_time = None
        self.history = None

        # Build the GRU model
        self.model = Sequential([
            Input(shape=input_shape),
            GRU(hidden_size),
            Dense(num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train):
        """
        Train the GRU model using an internal validation split.
        """

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

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

        actual_epochs = len(
            self.history.history["loss"]
        )

        if actual_epochs < self.epochs:
            print(
                f"\nEarly stopping triggered "
                f"at epoch {actual_epochs}"
            )

        print(
            f"GRU training completed in "
            f"{self.training_time:.4f} seconds"
        )

    def plot_history(self):
        """
        Plot training and validation loss and accuracy.
        """

        history = self.history.history
        epochs = range(
            1,
            len(history["loss"]) + 1
        )

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(12, 4)
        )

        # Loss curves
        ax1.plot(
            epochs,
            history["loss"],
            label="Training Loss"
        )

        ax1.plot(
            epochs,
            history["val_loss"],
            label="Validation Loss"
        )

        ax1.set_title(
            "Training vs Validation Loss"
        )

        ax1.set_xlabel("Epoch")
        ax1.legend()

        # Accuracy curves
        ax2.plot(
            epochs,
            history["accuracy"],
            label="Training Accuracy"
        )

        ax2.plot(
            epochs,
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        ax2.set_title(
            "Training vs Validation Accuracy"
        )

        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()

        plt.savefig(
            "../results/gru_same_hyperparameter_lstm.png",
            dpi=150
        )

        plt.close()

        print(
            "Training curves saved: "
            "gru_same_hyperparameter_lstm.png"
        )

    def predict(self, X_test):
        """
        Generate predictions for the test set.
        """

        start = time.time()

        probabilities = self.model.predict(
            X_test,
            verbose=0
        )

        self.inference_time = (
            time.time() - start
        )

        print(
            f"GRU inference completed in "
            f"{self.inference_time:.4f} seconds"
        )

        return np.argmax(
            probabilities,
            axis=1
        )

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and return accuracy
        and predicted labels.
        """

        predictions = self.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"GRU Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        return accuracy, predictions


if __name__ == "__main__":

    print("=" * 60)
    print("GRU Model")
    print("=" * 60)

    # Load dataset
    X, y, encoder = load_real_data()

    # Preprocess dataset
    X_train, X_test, y_train, y_test, scaler = preprocess(
        X,
        y
    )

    # Use the same hyperparameters as the LSTM model
    # to ensure a fair comparison.
    model = GRUModel(
        input_shape=(500, 256),
        hidden_size=64,
        num_classes=5,
        learning_rate=0.005,
        epochs=200,
        batch_size=32,
        patience=20
    )

    # Train model
    model.train(
        X_train,
        y_train
    )

    # Plot training history
    model.plot_history()

    # Evaluate model
    accuracy, predictions = model.evaluate(
        X_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    print(
        f"Test Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Training Time : "
        f"{model.training_time:.2f} seconds"
    )

    print(
        f"Inference Time: "
        f"{model.inference_time:.4f} seconds"
    )