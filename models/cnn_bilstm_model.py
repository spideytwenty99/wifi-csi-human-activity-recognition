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

"""
CNN + BiLSTM Kombination
========================
Erweiterung des CNN+LSTM Modells (Opt 9) durch ein bidirektionales LSTM.

Warum BiLSTM?
  Ein normales LSTM verarbeitet die Sequenz nur vorwärts (Vergangenheit → Zukunft).
  Ein BiLSTM fügt eine zweite LSTM-Schicht hinzu, die rückwärts läuft (Zukunft → Vergangenheit).
  Beide Ausgaben werden zusammengeführt → das Modell kennt für jeden Zeitschritt
  sowohl den vergangenen als auch den zukünftigen Kontext.

  Beispiel: Das Ende einer "Hinsetzen"-Bewegung gibt Kontext über den Anfang.
  Elkelany et al. (2023) zeigen, dass BiLSTM für WiFi-CSI-HAR besser abschneidet
  als unidirektionales LSTM, weil Aktivitäten zeitlich symmetrische Muster haben.

Unterschied zu CNN+LSTM (Opt 9):
  LSTM(64)              → Ausgabe: (64,)
  Bidirectional(LSTM(64)) → Ausgabe: (128,)  [64 vorwärts + 64 rückwärts]

  Der Dense-Layer bekommt automatisch die doppelte Eingabegröße — keine weitere
  Anpassung nötig. Die CNN-Schichten und alle Regularisierungsmaßnahmen bleiben
  identisch zu Opt 9.

Architektur-Übersicht:
  Input (500, 256)
  → Conv1D(128) + BatchNorm + MaxPool  → (249, 128)   grobe Muster
  → Dropout(0.3)
  → Conv1D(128) + BatchNorm + MaxPool  → (123, 128)   komplexere Muster
  → Dropout(0.3)
  → Conv1D(128) + BatchNorm + MaxPool  → (60, 128)    abstrakte Features
  → Dropout(0.3)
  → Bidirectional(LSTM(64))            → (128,)        vorwärts + rückwärts
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


class CNNBiLSTMModel:

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

            # ── CNN-Block 1 ────────────────────────────────────────────────────
            Conv1D(filters=filters, kernel_size=kernel_size,
                   activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),   # 500 → 249
            Dropout(dropout_rate),

            # ── CNN-Block 2 ────────────────────────────────────────────────────
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),   # 249 → 123
            Dropout(dropout_rate),

            # ── CNN-Block 3 ────────────────────────────────────────────────────
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),   # 123 → 60
            Dropout(dropout_rate),

            # ── BiLSTM: vorwärts + rückwärts ──────────────────────────────────
            # Bidirectional() hüllt das LSTM ein und fügt eine zweite Richtung hinzu.
            # Ausgabe: 64 (vorwärts) + 64 (rückwärts) = 128 Werte pro Sample
            # → mehr Kontext als unidirektionales LSTM, aber doppelter Hidden State
            Bidirectional(LSTM(hidden_size)),
            Dropout(dropout_rate),

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

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
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
        print(f"CNN+BiLSTM Training abgeschlossen in {self.training_time:.4f} Sekunden")


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
        plt.savefig('cnn_bilstm_training_curves.png', dpi=150)
        plt.close()
        print("Training Kurven gespeichert: cnn_bilstm_training_curves.png")


    ###############
    # Vorhersage
    ###############

    def predict(self, X_test):
        start = time.time()
        raw = self.model.predict(X_test, verbose=0)
        self.inference_time = time.time() - start
        print(f"CNN+BiLSTM Vorhersage abgeschlossen in {self.inference_time:.4f} Sekunden")
        return np.argmax(raw, axis=1)


    #################
    # Auswertung
    #################

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"CNN+BiLSTM Accuracy: {accuracy * 100:.2f}%")
        return accuracy, predictions
