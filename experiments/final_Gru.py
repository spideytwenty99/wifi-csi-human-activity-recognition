"""
Train and evaluate the final GRU model for activity classification
using CSI data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import random
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_real_data import load_real_data
from models.gru_model import GRUModel
from preprocessing.preprocess import preprocess


# Set random seeds for reproducibility
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


print("=" * 60)
print("GRU Model")
print("=" * 60)

# Load and preprocess the dataset
X, y, encoder = load_real_data()

X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

# Create the GRU model
model = GRUModel(
    input_shape=(500, 256),
    hidden_size=64,
    num_classes=5,
    learning_rate=0.005,
    epochs=50,
    batch_size=32,
    patience=20,
    dropout=0.2,
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

print("Train class counts:", np.bincount(y_train))
print("Test class counts :", np.bincount(y_test))

print("Train checksum:", np.sum(X_train))
print("Test checksum :", np.sum(X_test))

# Train the model
model.train(X_train, y_train)

# Plot training history from the model class
model.plot_history()

# Evaluate the model
accuracy, predictions = model.evaluate(
    X_test,
    y_test
)

print("\n" + "=" * 60)
print("Final Results")
print("=" * 60)

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Training Time : {model.training_time:.2f} seconds")
print(f"Inference Time: {model.inference_time:.4f} seconds")

history = model.history.history

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Generate and display the confusion matrix
cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

plt.figure(figsize=(7, 6))

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("GRU Confusion Matrix")
plt.tight_layout()
plt.show()