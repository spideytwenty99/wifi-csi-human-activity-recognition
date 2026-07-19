import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

"""
1D CNN (Convolutional Neural Network)
Ein CNN verwendet Filter (auch Kernel genannt), die über die Eingabe gleiten und lokale Muster suchen.
Bei Zeitreihendaten (1D CNN) schaut ein Filter z.B. 3 oder 5 aufeinanderfolgende Zeitschritte an
und sucht nach charakteristischen Mustern — z.B. dem Schwingungsrhythmus beim Gehen.

Architektur:
  Conv1D   → Filter gleiten über Zeitschritte, erkennen lokale Muster
  MaxPool  → Verkleinert die Feature Map, behält nur den stärksten Wert pro Fenster
  Flatten  → Wandelt 2D Feature Maps in 1D Vektor um
  Dense    → Vollverbundene Schicht für die finale Klassifikation
  Dropout  → Schaltet zufällig 30% der Neuronen ab (verhindert Overfitting)
  Softmax  → Gibt Wahrscheinlichkeiten für jede Klasse aus
"""

tf.random.set_seed(42)


class CNNModel:

    def __init__(self, input_shape=(500, 256), num_classes=5,
                 filters=128, kernel_size=3, learning_rate=0.0005,
                 epochs=30, batch_size=16):

        self.epochs = epochs
        self.batch_size = batch_size
        self.training_time = None
        self.inference_time = None
        self.history = None

        # Modell aufbauen
        self.model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                   input_shape=input_shape),   # erster Faltungsblock
            MaxPooling1D(pool_size=2),          # halbiert die Zeitdimension

            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),          # zweiter Faltungsblock

            Flatten(),                          # 2D → 1D
            Dense(128, activation='relu'),      # vollverbundene Schicht
            Dropout(0.3),                       # 30% Neuronen zufällig deaktivieren
            Dense(num_classes, activation='softmax')  # Ausgabe: Wahrscheinlichkeit pro Klasse
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
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

        start = time.time()
        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1  # Keras gibt jeden Epoch aus (nur 30 Epochen, übersichtlich)
        )
        self.training_time = time.time() - start
        print(f"CNN Training abgeschlossen in {self.training_time:.4f} Sekunden")


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
        plt.savefig('cnn_training_curves.png', dpi=150)
        plt.close()
        print("Training Kurven gespeichert: cnn_training_curves.png")


    ###############
    # Vorhersage
    ###############

    def predict(self, X_test):
        start = time.time()
        raw = self.model.predict(X_test, verbose=0)
        self.inference_time = time.time() - start
        print(f"CNN Vorhersage abgeschlossen in {self.inference_time:.4f} Sekunden")
        return np.argmax(raw, axis=1)


    #################
    # Auswertung
    #################

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"CNN Accuracy: {accuracy * 100:.2f}%")
        return accuracy, predictions
