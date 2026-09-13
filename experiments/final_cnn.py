import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_real_data import load_real_data
from preprocessing.preprocess import preprocess
from models.cnn_model import CNNModel


print("=" * 60)
print("CNN Model")
print("=" * 60)

# Load dataset
X, y, encoder = load_real_data()

print(f"Dataset Shape : {X.shape}")
print(f"Labels Shape  : {y.shape}")
print(f"Classes       : {encoder.classes_}")

# Preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# Create model
model = CNNModel(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    num_classes=len(encoder.classes_),
    filters=128,
    kernel_size=3,
    learning_rate=0.0005,
    epochs=30,
    batch_size=16
)

# Train model
model.train(X_train, y_train)

# Plot training history
history = model.history.history

plt.figure(figsize=(12, 5))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=300)
plt.show()

# Generate predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Training Time : {model.training_time:.4f} seconds")
print(f"Inference Time: {model.inference_time:.4f} seconds")
print("=" * 60)

# Plot confusion matrix
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png", dpi=300)
plt.show()