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

"""
LSTM (Long Short-Term Memory)
Ein normales Perceptron hat kein Gedächtnis — jede Eingabe wird unabhängig behandelt.
Bei Zeitreihen ist das ein Problem: Um Gehen zu erkennen, braucht man nicht nur den
aktuellen Signalwert, sondern auch die Werte der letzten Zeitschritte.

LSTM hat 2 Gedächtnisse:
  Cell State   → Langzeitgedächtnis (geht durch die ganze Sequenz)
  Hidden State → Kurzzeitgedächtnis (Ausgabe des aktuellen Zeitschritts)

Und drei Gates die kontrollieren was erinnert und vergessen wird:
  Forget Gate → Welche alten Informationen werden gelöscht?
  Input Gate  → Welche neuen Informationen werden gespeichert?
  Output Gate → Was wird an den nächsten Zeitschritt weitergegeben?
"""

tf.random.set_seed(42)


class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    """Gibt Training-Metriken alle 10 Epochen aus — während des Trainings."""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            total = self.params['epochs']
            print(f"Epoch {epoch+1}/{total} - "
                  f"Train Loss: {logs['loss']:.4f} | "
                  f"Val Loss: {logs['val_loss']:.4f} | "
                  f"Train Acc: {logs['accuracy']*100:.1f}% | "
                  f"Val Acc: {logs['val_accuracy']*100:.1f}%")


class LSTMModel:

    def __init__(self, input_shape=(500, 256), hidden_size=64, num_classes=5,
                 learning_rate=0.005, epochs=200, batch_size=32, patience=20):

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.training_time = None
        self.inference_time = None
        self.history = None

        # Modell aufbauen
        self.model = Sequential([
            LSTM(hidden_size, input_shape=input_shape),  # LSTM-Schicht mit 64 Neuronen
            Dense(num_classes, activation='softmax')     # Ausgabeschicht: Score pro Klasse
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),  # Adam passt Gewichte an
            loss='sparse_categorical_crossentropy',        # Verlustfunktion für Klassifikation
            metrics=['accuracy']
        )


    #############
    # Training
    #############

    def train(self, X_train, y_train):
        # 20% der Trainingsdaten als Validierungsset
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True  # stellt die besten Gewichte wieder her
        )

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

        actual_epochs = len(self.history.history['loss'])
        if actual_epochs < self.epochs:
            print(f"\nEarly Stopping bei Epoch {actual_epochs}")
        print(f"LSTM Training abgeschlossen in {self.training_time:.4f} Sekunden")


    #################
    # History plotten
    #################

    def plot_history(self):
        h = self.history.history
        epochs = range(1, len(h['loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, h['loss'],     label='Train Loss')
        ax1.plot(epochs, h['val_loss'], label='Val Loss')
        ax1.set_title('Train Loss vs Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        ax2.plot(epochs, h['accuracy'],     label='Train Acc')
        ax2.plot(epochs, h['val_accuracy'], label='Val Acc')
        ax2.set_title('Train Accuracy vs Val Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('lstm_training_curves.png', dpi=150)
        plt.close()
        print("Training Kurven gespeichert: lstm_training_curves.png")


    ###############
    # Vorhersage
    ###############

    def predict(self, X_test):
        start = time.time()
        raw = self.model.predict(X_test, verbose=0)  # gibt Wahrscheinlichkeiten pro Klasse zurück
        self.inference_time = time.time() - start
        print(f"LSTM Vorhersage abgeschlossen in {self.inference_time:.4f} Sekunden")
        return np.argmax(raw, axis=1)  # Index mit höchster Wahrscheinlichkeit = vorhergesagte Klasse


    #################
    # Auswertung
    #################

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"LSTM Accuracy: {accuracy * 100:.2f}%")
        return accuracy, predictions
