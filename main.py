"""
Model Comparison Script for HAR (Human Activity Recognition)

This script compares all 6 models on the WiFi-CSI dataset:
1. Data loading → Preprocessing → Feature extraction
2. Train and test all models
3. Compare results (accuracy + runtime)
4. Display comparison table and visualizations

Models compared:
- k-NN (k-Nearest Neighbors)
- SVM (Support Vector Machine)
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- CNN+LSTM (Hybrid)
- CNN+BiLSTM (Hybrid with bidirectional LSTM)
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.load_real_data import load_real_data
from preprocessing.preprocess import preprocess
from features.feature_extraction import extract_features
from models.knn_model import KNNModel
from models.svm_model import SVMModel
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.cnn_lstm_model import CNNLSTMModel
from models.cnn_bilstm_model import CNNBiLSTMModel
from evaluation.evaluate import evaluate_all


# ============================================================
# Data Preparation
# ============================================================

print("=" * 50)
print("  WiFi Activity Recognition - Model Comparison")
print("=" * 50)

print("\n[1] Loading dataset...")
X, y, encoder = load_real_data()
print(f"    Dataset: {X.shape[0]} samples, {X.shape[1]} timesteps")

print("\n[2] Preprocessing...")
X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
print(f"    Training: {X_train.shape[0]} samples")
print(f"    Test:     {X_test.shape[0]} samples")

print("\n[3] Feature Extraction (for k-NN and SVM)...")
F_train = extract_features(X_train)
F_test = extract_features(X_test)
print(f"    Feature Shape: {F_train.shape}")


# ============================================================
# Train and Evaluate All Models
# ============================================================

results = {}

# k-NN Model
print("\n[4] k-NN Training & Evaluation...")
knn = KNNModel(k=5, metric='manhattan')
knn.train(F_train, y_train)
acc_knn, pred_knn = knn.evaluate(F_test, y_test, F_train, y_train)
results['k-NN'] = {
    'accuracy': acc_knn,
    'training_time': knn.training_time,
    'inference_time': knn.inference_time,
    'predictions': pred_knn
}

# SVM Model
print("\n[5] SVM Training & Evaluation...")
svm = SVMModel()
svm.train(F_train, y_train)
acc_svm, pred_svm = svm.evaluate(F_test, y_test)
results['SVM'] = {
    'accuracy': acc_svm,
    'training_time': svm.training_time,
    'inference_time': svm.inference_time,
    'predictions': pred_svm
}

# LSTM Model
print("\n[6] LSTM Training & Evaluation...")
lstm = LSTMModel(epochs=200)
lstm.train(X_train, y_train)
acc_lstm, pred_lstm = lstm.evaluate(X_test, y_test)
results['LSTM'] = {
    'accuracy': acc_lstm,
    'training_time': lstm.training_time,
    'inference_time': lstm.inference_time,
    'predictions': pred_lstm
}
lstm.plot_history()

# CNN Model
print("\n[7] CNN Training & Evaluation...")
cnn = CNNModel(epochs=30)
cnn.train(X_train, y_train)
acc_cnn, pred_cnn = cnn.evaluate(X_test, y_test)
results['CNN'] = {
    'accuracy': acc_cnn,
    'training_time': cnn.training_time,
    'inference_time': cnn.inference_time,
    'predictions': pred_cnn
}
cnn.plot_history()

# CNN+LSTM Model
print("\n[8] CNN+LSTM Training & Evaluation...")
cnn_lstm = CNNLSTMModel(epochs=100)
cnn_lstm.train(X_train, y_train)
acc_cnn_lstm, pred_cnn_lstm = cnn_lstm.evaluate(X_test, y_test)
results['CNN+LSTM'] = {
    'accuracy': acc_cnn_lstm,
    'training_time': cnn_lstm.training_time,
    'inference_time': cnn_lstm.inference_time,
    'predictions': pred_cnn_lstm
}
cnn_lstm.plot_history()

# CNN+BiLSTM Model
print("\n[9] CNN+BiLSTM Training & Evaluation...")
cnn_bilstm = CNNBiLSTMModel(epochs=100)
cnn_bilstm.train(X_train, y_train)
acc_cnn_bilstm, pred_cnn_bilstm = cnn_bilstm.evaluate(X_test, y_test)
results['CNN+BiLSTM'] = {
    'accuracy': acc_cnn_bilstm,
    'training_time': cnn_bilstm.training_time,
    'inference_time': cnn_bilstm.inference_time,
    'predictions': pred_cnn_bilstm
}
cnn_bilstm.plot_history()


# ============================================================
# Display Comparison Results
# ============================================================

print("\n")
print("=" * 55)
print("  RESULTS - MODEL COMPARISON")
print("=" * 55)
print(f"{'Model':<12} {'Accuracy':>10} {'Training':>12} {'Inference':>12}")
print("-" * 55)

for name, r in results.items():
    print(f"{name:<12} "
          f"{r['accuracy']*100:>9.2f}% "
          f"{r['training_time']:>11.4f}s "
          f"{r['inference_time']:>11.4f}s")

print("=" * 55)

# Find and display the best performing model
best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n  Best Model: {best_model} "
      f"({results[best_model]['accuracy']*100:.2f}% Accuracy)")
print("=" * 55)


# ============================================================
# Generate Evaluation Visualizations
# ============================================================

class_names = encoder.classes_.tolist()
evaluate_all(results, y_test, class_names)