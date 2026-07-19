import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from data.load_real_data import load_real_data
from models.gru_model import GRUModel
from preprocessing.preprocess import preprocess
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("=" * 60)
print("                  GRU Model")
print("=" * 60)
# Load dataset
X, y, encoder = load_real_data()

X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

# Create model
model = GRUModel(
    input_shape=(500, 256),
    hidden_size=64,
    num_classes=5,
    learning_rate=0.005,
    epochs=200,
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

# Train
model.train(X_train, y_train)

# Plot training curves
model.plot_history()

# Evaluate
accuracy, predictions = model.evaluate(X_test, y_test)

print("\n" + "=" * 60)
print("Final Results")
print("=" * 60)
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Training Time : {model.training_time:.2f} seconds")
print(f"Inference Time: {model.inference_time:.4f} seconds")