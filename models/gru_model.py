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

"""
GRU (Gated Recurrent Unit)

Wie LSTM ist GRU für Zeitreihen entwickelt.
Im Gegensatz zu einem normalen neuronalen Netz kann GRU
Informationen über mehrere Zeitschritte hinweg speichern.

GRU besitzt nur einen Hidden State und verwendet zwei Gates:

Update Gate → Entscheidet, welche Informationen behalten werden.
Reset Gate  → Entscheidet, welche alten Informationen vergessen werden.

Dadurch besitzt GRU weniger Parameter als LSTM und trainiert
häufig schneller bei vergleichbarer Genauigkeit.
"""

class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    """Gibt Training-Metriken alle 10 Epochen aus."""

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            total = self.params["epochs"]

            print(
                f"Epoch {epoch+1}/{total} - "
                f"Train Loss: {logs['loss']:.4f} | "
                f"Val Loss: {logs['val_loss']:.4f} | "
                f"Train Acc: {logs['accuracy']*100:.1f}% | "
                f"Val Acc: {logs['val_accuracy']*100:.1f}%"
            )


class GRUModel:

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

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dropout=dropout

        self.training_time = None
        self.inference_time = None

        self.history = None

        self.model = Sequential([
            Input(shape=input_shape),
            GRU(128,dropout=self.dropout, return_sequences=True), # added return sequnce and the extra layer down
            GRU(hidden_size,dropout=self.dropout),
            Dense(num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )


    #################
    # Training
    #################

    def train(self, X_train, y_train):

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
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

        actual_epochs = len(self.history.history["loss"])

        if actual_epochs < self.epochs:
            print(f"\nEarly Stopping at Epoch {actual_epochs}")

        print(
            f"GRU Training completed in "
            f"{self.training_time:.4f} seconds"
        )

    #################
    # Plot History
    #################

    def plot_history(self):

        h = self.history.history

        epochs = range(1, len(h["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, h["loss"], label="Train Loss")
        ax1.plot(epochs, h["val_loss"], label="Validation Loss")
        ax1.set_title("Train Loss vs Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs, h["accuracy"], label="Train Accuracy")
        ax2.plot(epochs, h["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Train Accuracy vs Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("gru_training_curves.png", dpi=150)
        plt.close()

        print("Training curves saved as gru_training_curves.png")

    #################
    # Prediction
    #################

    def predict(self, X_test):

        start = time.time()

        probabilities = self.model.predict(
            X_test,
            verbose=0
        )

        self.inference_time = time.time() - start

        print(
            f"GRU Prediction completed in "
            f"{self.inference_time:.4f} seconds"
        )

        return np.argmax(probabilities, axis=1)

    #################
    # Evaluation
    #################

    def evaluate(self, X_test, y_test):

        predictions = self.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(f"GRU Accuracy: {accuracy * 100:.2f}%")

        return accuracy, predictions