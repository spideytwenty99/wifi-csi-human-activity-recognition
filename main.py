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


"""
Vergleich aller 6 Modelle:
Daten laden → Preprocessing → Features extrahieren
→ Alle Modelle trainieren & testen
→ Ergebnisse vergleichen (Accuracy + Laufzeit)
→ Tabelle ausgeben
"""


####################
# Daten vorbereiten
####################

print("=" * 50)
print("  WiFi Aktivitätserkennung - Modellvergleich")
print("=" * 50)

print("\n[1] Daten generieren...")
X, y, encoder = load_real_data()
print(f"    Datensatz: {X.shape[0]} Samples, {X.shape[1]} Zeitschritte")

print("\n[2] Preprocessing...")
X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
print(f"    Training: {X_train.shape[0]} Samples")
print(f"    Test:     {X_test.shape[0]} Samples")

print("\n[3] Feature Extraction (für k-NN und SVM)...")
F_train = extract_features(X_train)
F_test  = extract_features(X_test)
print(f"    Feature Shape: {F_train.shape}")


##############################
# Alle Modelle trainieren und testen
##############################

results = {}

k-NN
print("\n[4] k-NN Training & Evaluation...")
knn = KNNModel(k=5, metric='manhattan')
knn.train(F_train, y_train)
acc_knn, pred_knn = knn.evaluate(F_test, y_test, F_train, y_train)
results['k-NN'] = {
    'accuracy':       acc_knn,
    'training_time':  knn.training_time,
    'inference_time': knn.inference_time,
    'predictions':    pred_knn
}


# LSTM
print("\n[6] LSTM Training & Evaluation...")
lstm = LSTMModel(epochs=200)
lstm.train(X_train, y_train)
acc_lstm, pred_lstm = lstm.evaluate(X_test, y_test)
results['LSTM'] = {
    'accuracy':       acc_lstm,
    'training_time':  lstm.training_time,
    'inference_time': lstm.inference_time,
    'predictions':    pred_lstm
}
lstm.plot_history()

# CNN
print("\n[7] CNN Training & Evaluation...")
cnn = CNNModel(epochs=30)
cnn.train(X_train, y_train)
acc_cnn, pred_cnn = cnn.evaluate(X_test, y_test)
results['CNN'] = {
    'accuracy':       acc_cnn,
    'training_time':  cnn.training_time,
    'inference_time': cnn.inference_time,
    'predictions':    pred_cnn
}
cnn.plot_history()

# CNN+LSTM
print("\n[8] CNN+LSTM Training & Evaluation...")
cnn_lstm = CNNLSTMModel(epochs=100)
cnn_lstm.train(X_train, y_train)
acc_cnn_lstm, pred_cnn_lstm = cnn_lstm.evaluate(X_test, y_test)
results['CNN+LSTM'] = {
    'accuracy':       acc_cnn_lstm,
    'training_time':  cnn_lstm.training_time,
    'inference_time': cnn_lstm.inference_time,
    'predictions':    pred_cnn_lstm
}
cnn_lstm.plot_history()

# CNN+BiLSTM
print("\n[9] CNN+BiLSTM Training & Evaluation...")
cnn_bilstm = CNNBiLSTMModel(epochs=100)
cnn_bilstm.train(X_train, y_train)
acc_cnn_bilstm, pred_cnn_bilstm = cnn_bilstm.evaluate(X_test, y_test)
results['CNN+BiLSTM'] = {
    'accuracy':       acc_cnn_bilstm,
    'training_time':  cnn_bilstm.training_time,
    'inference_time': cnn_bilstm.inference_time,
    'predictions':    pred_cnn_bilstm
}
cnn_bilstm.plot_history()




#########################
# Vergleichstabelle ausgeben
#########################

print("\n")
print("=" * 55)
print("  ERGEBNISSE - MODELLVERGLEICH")
print("=" * 55)
print(f"{'Modell':<10} {'Accuracy':>10} {'Training':>12} {'Inference':>12}")
print("-" * 55)

for name, r in results.items():
    print(f"{name:<10} "
          f"{r['accuracy']*100:>9.2f}% "
          f"{r['training_time']:>11.4f}s "
          f"{r['inference_time']:>11.4f}s")

print("=" * 55)

best = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n  Bestes Modell: {best} "
      f"({results[best]['accuracy']*100:.2f}% Accuracy)")
print("=" * 55)


class_names = encoder.classes_.tolist()
evaluate_all(results, y_test, class_names)
