"""
K-NN Model Module for HAR (Human Activity Recognition)

This module implements a k-Nearest Neighbors classifier for human activity
recognition from extracted features.

k-NN (k-Nearest Neighbors) Overview:
The idea is simple: when a new sample needs to be classified, k-NN looks at
the k training samples that are most similar and takes the most common class
among them.

Example with k=3:
New sample arrives
→ The 3 most similar training samples are: Class 1, Class 1, Class 2
→ Class 1 wins (2 out of 3)
→ Prediction: Class 1
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


class KNNModel:
    """
    K-Nearest Neighbors classifier for HAR classification.

    This model uses distance-based similarity to classify new samples
    based on the majority class of their k nearest neighbors.
    """

    def __init__(self, k=5, metric='manhattan'):
        """
        Initialize the K-NN model with specified hyperparameters.

        Args:
            k (int): Number of nearest neighbors to consider
            metric (str): Distance metric ('euclidean' or 'manhattan')
        """
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        self.training_time = None
        self.inference_time = None

    def train(self, X_train, y_train):
        """
        Train the K-NN model on the provided data.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        start = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start

        print(f"K-NN Training completed in {self.training_time:.4f} seconds")

    def predict(self, X_test):
        """
        Generate predictions for test data.

        Args:
            X_test (np.ndarray): Test features

        Returns:
            np.ndarray: Predicted class labels
        """
        start = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start

        print(f"K-NN Inference completed in {self.inference_time:.4f} seconds")

        return predictions

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        """
        Evaluate model performance on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True labels
            X_train (np.ndarray, optional): Training features for train accuracy
            y_train (np.ndarray, optional): Training labels for train accuracy

        Returns:
            tuple: (accuracy, predictions)
                - accuracy (float): Test accuracy as a fraction
                - predictions (np.ndarray): Predicted class labels
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"K-NN Accuracy: {accuracy * 100:.2f}%")

        # Calculate training accuracy if training data is provided
        if X_train is not None:
            train_predictions = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)
            print(f"K-NN Train Accuracy: {train_accuracy * 100:.2f}%")

        return accuracy, predictions


# Test the K-NN model with different hyperparameters
if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.load_real_data import load_real_data
    from preprocessing.preprocess import preprocess
    from features.feature_extraction import extract_features
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # Load and preprocess data
    X, y, encoder = load_real_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

    # Extract features
    F_train = extract_features(X_train)
    F_test = extract_features(X_test)

    # Test different hyperparameter combinations
    print(f"\n{'k':>4} {'Metric':<12} {'Accuracy':>10}")
    print("-" * 30)

    for metric in ['euclidean', 'manhattan']:
        for k in [1, 3, 5, 7, 10, 15]:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
            clf.fit(F_train, y_train)
            acc = accuracy_score(y_test, clf.predict(F_test))
            print(f"{k:>4} {metric:<12} {acc * 100:>9.2f}%")