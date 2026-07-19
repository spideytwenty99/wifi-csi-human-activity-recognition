import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""
CNN + LSTM Kombination (Schritt 1: Vertiefte CNN-Architektur)
=============================================================
Verbesserungen gegenüber der ersten Version:

  1. Drei CNN-Blöcke statt zwei
     → Tiefere Merkmalhierarchie: grobe → komplexe → abstrakte Muster
     → Belegt durch Shang et al. (2021): tiefere CNN-Stufen verbessern
       die Feature-Qualität bei WiFi-CSI-Daten

  2. BatchNormalization nach jedem Conv1D
     → Normalisiert Aktivierungen auf Mittelwert≈0, Varianz≈1
     → Stabilisiert Training, verhindert explodierende/verschwindende Gradienten
     → Erlaubt schnellere Konvergenz

  3. Dropout(0.3) nach jedem CNN-Block und nach LSTM
     → Zufällig 30% der Neuronen pro Trainingsschritt deaktivieren
     → Netz kann sich nicht auf einzelne Neuronen verlassen → weniger Overfitting
     → Besonders wichtig bei kleinen Datensätzen (~960 Trainingssamples)

  4. ReduceLROnPlateau Callback
     → Lernrate wird halbiert wenn val_loss 5 Epochen stagniert
     → Feinere Anpassung der Gewichte in späteren Trainingsphasen

Architektur-Übersicht:
  Input (500, 256)
  → Conv1D(128) + BatchNorm + MaxPool  → (249, 128)   grobe Muster
  → Dropout(0.3)
  → Conv1D(128) + BatchNorm + MaxPool  → (123, 128)   komplexere Muster
  → Dropout(0.3)
  → Conv1D(128) + BatchNorm + MaxPool  → (60, 128)    abstrakte Hochlevel-Features
  → Dropout(0.3)
  → LSTM(64)                           → (64,)         zeitliche Abhängigkeiten
  → Dropout(0.3)
  → Dense(5, softmax)                  → Vorhersage
"""

tf.random.set_seed(42)


class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    """Gibt Training-Metriken alle 10 Epochen aus — während des Trainings."""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            total = self.params['epochs']
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            print(f"Epoch {epoch+1}/{total} - "
                  f"Train Loss: {logs['loss']:.4f} | "
                  f"Val Loss: {logs['val_loss']:.4f} | "
                  f"Train Acc: {logs['accuracy']*100:.1f}% | "
                  f"Val Acc: {logs['val_accuracy']*100:.1f}% | "
                  f"LR: {lr:.6f}")


class CNNLSTMModel:

    def __init__(self, input_shape=(500, 256), num_classes=5,
                 filters=128, kernel_size=3, hidden_size=64,
                 dropout_rate=0.3,
                 learning_rate=0.0005, epochs=100, batch_size=32, patience=20):

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.training_time = None
        self.inference_time = None
        self.history = None

        self.model = Sequential([

            # ── CNN-Block 1: Grobe Merkmale aus rohen Zeitreihen ──────────────
            # 128 Filter der Breite 3 gleiten über die 500 Zeitschritte
            # → erkennt kurze, lokale Muster (z.B. plötzliche Amplitudenänderung)
            Conv1D(filters=filters, kernel_size=kernel_size,
                   activation='relu', input_shape=input_shape),
            BatchNormalization(),        # Aktivierungen normalisieren → stabileres Training
            MaxPooling1D(pool_size=2),   # 500 → 249 Zeitschritte (Dimensionsreduktion)
            Dropout(dropout_rate),       # 30% Neuronen zufällig deaktivieren → weniger Overfitting

            # ── CNN-Block 2: Komplexere Muster ────────────────────────────────
            # Verarbeitet die 249 komprimierten Features aus Block 1
            # → erkennt längere Muster (z.B. Rhythmus einer Gehbewegung)
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),   # 249 → 123 Zeitschritte
            Dropout(dropout_rate),

            # ── CNN-Block 3: Abstrakte Hochlevel-Features ─────────────────────
            # Dritter Block ermöglicht tiefere Merkmal-Hierarchie
            # → Shang et al. (2021) zeigen: mehr CNN-Stufen verbessern
            #   WiFi-CSI-Klassifikation durch reichhaltigere Repräsentationen
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),   # 123 → 60 Zeitschritte
            Dropout(dropout_rate),

            # ── LSTM: Zeitliche Abhängigkeiten modellieren ────────────────────
            # Verarbeitet nun nur noch 60 kompakte Feature-Vektoren (statt 500 Rohdaten)
            # → viel effizienter und stabiler als direktes LSTM auf Rohdaten
            LSTM(hidden_size),
            Dropout(dropout_rate),       # Auch nach LSTM regularisieren

            # ── Ausgabeschicht ─────────────────────────────────────────────────
            Dense(num_classes, activation='softmax')
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
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # EarlyStopping: stoppt wenn val_loss sich 20 Epochen nicht verbessert
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True  # beste Gewichte wiederherstellen
        )

        # ReduceLROnPlateau: halbiert Lernrate wenn val_loss 5 Epochen stagniert
        # → hilft dem Modell sich in späteren Phasen feiner einzupendeln
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,       # Lernrate × 0.5 (halbieren)
            patience=5,       # nach 5 Epochen ohne Verbesserung
            min_lr=1e-6,      # Untergrenze — Lernrate wird nie kleiner als das
            verbose=0
        )

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

        actual_epochs = len(self.history.history['loss'])
        if actual_epochs < self.epochs:
            print(f"\nEarly Stopping bei Epoch {actual_epochs}")
        print(f"CNN+LSTM Training abgeschlossen in {self.training_time:.4f} Sekunden")


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
        plt.savefig('cnn_lstm_training_curves.png', dpi=150)
        plt.close()
        print("Training Kurven gespeichert: cnn_lstm_training_curves.png")


    ###############
    # Vorhersage
    ###############

    def predict(self, X_test):
        start = time.time()
        raw = self.model.predict(X_test, verbose=0)
        self.inference_time = time.time() - start
        print(f"CNN+LSTM Vorhersage abgeschlossen in {self.inference_time:.4f} Sekunden")
        return np.argmax(raw, axis=1)


    #################
    # Auswertung
    #################

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"CNN+LSTM Accuracy: {accuracy * 100:.2f}%")
        return accuracy, predictions
