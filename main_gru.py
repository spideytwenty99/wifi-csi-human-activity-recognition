# import itertools
# import gc
# import pandas as pd
# import tensorflow as tf
# from numpy import random
# import numpy as np
# from data.load_real_data import load_real_data
# from preprocessing.preprocess import preprocess
# from models.gru_model import GRUModel
#
#
# SEED = 42
#
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
#
# ####################
# # Load Dataset
# ####################
#
# print("=" * 60)
# print("          GRU Hyperparameter Tuning")
# print("=" * 60)
#
# X, y, encoder = load_real_data()
#
# X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
#
#
# ####################
# # Hyperparameters
# ####################
#
# hidden_sizes = [64]
# learning_rates = [0.005]
# batch_sizes = [16, 32, 64]
# dropouts = [0.0, 0.1, 0.2]
# patiences = [20, 30]
#
#
# ####################
# # Variables
# ####################
#
# results = []
#
# total = (
#     len(hidden_sizes)
#     * len(learning_rates)
#     * len(batch_sizes)
#     * len(dropouts)
#     * len(patiences)
# )
#
# experiment = 1
#
#
# ####################
# # Grid Search
# ####################
#
# for hidden_size, lr, batch_size, dropout, patience in itertools.product(
#         hidden_sizes,
#         learning_rates,
#         batch_sizes,
#         dropouts,
#         patiences
# ):
#
#     print("\n" + "=" * 60)
#     print(f"Experiment {experiment}/{total}")
#     print(f"Hidden Size   : {hidden_size}")
#     print(f"Learning Rate : {lr}")
#     print(f"Batch Size    : {batch_size}")
#     print(f"Dropout       : {dropout}")
#     print(f"Patience      : {patience}")
#
#     model = GRUModel(
#         hidden_size=hidden_size,
#         learning_rate=lr,
#         batch_size=batch_size,
#         epochs=200,
#         patience=patience,
#         dropout=dropout
#     )
#
#     model.train(X_train, y_train)
#
#     accuracy, _ = model.evaluate(X_test, y_test)
#
#     results.append({
#         "Hidden Size": hidden_size,
#         "Learning Rate": lr,
#         "Batch Size": batch_size,
#         "Dropout": dropout,
#         "Patience": patience,
#         "Accuracy": accuracy,
#         "Training Time (s)": round(model.training_time, 2),
#         "Inference Time (s)": round(model.inference_time, 2)
#     })
#
#     # Free memory before next experiment
#     del model
#     tf.keras.backend.clear_session()
#     gc.collect()
#
#     experiment += 1
#
#
# ####################
# # Results
# ####################
#
# results = pd.DataFrame(results)
#
# results = results.sort_values(
#     by="Accuracy",
#     ascending=False
# )
#
# print("\n")
# print("=" * 60)
# print("Top Configurations")
# print("=" * 60)
#
# print(results)
#
# # Optional: Save results
# results.to_csv("gru_hyperparameter_results.csv", index=False)
#
# print("\nResults saved to gru_hyperparameter_results.csv")

import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())