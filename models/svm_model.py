"""
SVM Model Module for HAR (Human Activity Recognition)

This module implements an SVM (Support Vector Machine) classifier with
hyperparameter tuning using GridSearchCV for human activity recognition
from extracted features.
"""

import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class SVMModel:
    """
    SVM Classifier with GridSearchCV hyperparameter optimization.

    This model uses a Support Vector Machine with RBF kernel by default
    and performs 5-fold cross-validation to find optimal hyperparameters.
    """

    def __init__(self):
        """
        Initialize the SVM model with default parameters.
        """
        self.model = None
        self.training_time = None
        self.inference_time = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.cv_accuracy = None

    def train(self, X_train, y_train):
        """
        Train the SVM model using GridSearchCV for hyperparameter tuning.

        Hyperparameter search space:
            - kernel: ['linear', 'rbf']
            - C: [0.1, 1, 10, 100, 1000] (regularization parameter)
            - gamma: ['scale', 0.001, 0.01, 0.1] (kernel coefficient)

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        # Define hyperparameter search space
        parameters = {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["scale", 0.001, 0.01, 0.1]
        }

        # Initialize GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(
            estimator=SVC(random_state=42),
            param_grid=parameters,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,  # Use all available CPU cores
            verbose=2  # Display progress
        )

        print("Starting Grid Search with 5-Fold Cross Validation...")

        # Perform grid search
        start = time.time()
        grid_search.fit(X_train, y_train)
        self.training_time = time.time() - start

        # Store the best model and its cross-validation accuracy
        self.model = grid_search.best_estimator_
        self.cv_accuracy = grid_search.best_score_

        print(f"\nTraining completed in {self.training_time:.4f} seconds")
        print("\nBest Hyperparameters:")
        print(grid_search.best_params_)
        print(f"\nBest Cross Validation Accuracy: {self.cv_accuracy:.4f}")

        # Fixed hyperparameters approach (commented for reference)
        # self.model = SVC(
        #     kernel="rbf",
        #     C=1.0,
        #     random_state=42
        # )
        # print("Training SVM with fixed hyperparameters...")
        # start = time.time()
        # self.model.fit(X_train, y_train)
        # self.training_time = time.time() - start
        # print(f"\nTraining completed in {self.training_time:.4f} seconds")

        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        self.train_accuracy = accuracy_score(y_train, train_predictions)
        print(f"\nTraining Accuracy : {self.train_accuracy:.4f}")

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

        print(f"SVM Prediction completed in {self.inference_time:.6f} seconds")

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True labels

        Returns:
            tuple: (accuracy, predictions)
                - accuracy (float): Test accuracy as a fraction
                - predictions (np.ndarray): Predicted class labels
        """
        # Generate predictions
        predictions = self.predict(X_test)

        # Calculate test accuracy
        self.test_accuracy = accuracy_score(y_test, predictions)

        # Display results
        print(f"\nTest Accuracy     : {self.test_accuracy:.4f}")
        print(f"Difference        : {self.train_accuracy - self.test_accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        return self.test_accuracy, predictions