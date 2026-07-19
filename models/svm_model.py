import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class SVMModel:

    def __init__(self):

        self.model = None

        self.training_time = None
        self.inference_time = None

        self.train_accuracy = None
        self.test_accuracy = None
        self.cv_accuracy = None


    ##############################
    # Training
    ##############################

    def train(self, X_train, y_train):

        parameters = {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["scale", 0.001, 0.01, 0.1]
        }

        grid_search = GridSearchCV(
            estimator=SVC(random_state=42),
            param_grid=parameters,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2
        )

        print("Starting Grid Search with 5-Fold Cross Validation...")

        start = time.time()

        grid_search.fit(
            X_train,
            y_train
        )

        self.training_time = time.time() - start

        self.model = grid_search.best_estimator_

        self.cv_accuracy = grid_search.best_score_

        print(f"\nTraining completed in {self.training_time:.4f} seconds")

        print("\nBest Hyperparameters")
        print(grid_search.best_params_)

        #fixed-hyperpara-noGridSerach
        # self.model = SVC(
        #     kernel="rbf",
        #     C=1.0,
        #     random_state=42
        # )
        #
        # print("Training SVM with fixed hyperparameters...")
        #
        # start = time.time()
        #
        # self.model.fit(X_train, y_train)
        #
        # self.training_time = time.time() - start
        #
        # print(f"\nTraining completed in {self.training_time:.4f} seconds")
        #Fixed

        print(f"\nBest Cross Validation Accuracy: {self.cv_accuracy:.4f}")


        train_predictions = self.model.predict(X_train)

        self.train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        print(f"\nTraining Accuracy : {self.train_accuracy:.4f}")


    ##############################
    # Prediction
    ##############################

    def predict(self, X_test):

        start = time.time()

        predictions = self.model.predict(X_test)

        self.inference_time = time.time() - start

        print(
            f"SVM Prediction completed in {self.inference_time:.6f} seconds"
        )

        return predictions


    ##############################
    # Evaluation
    ##############################

    def evaluate(self, X_test, y_test):

        predictions = self.predict(X_test)

        self.test_accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(f"\nTest Accuracy     : {self.test_accuracy:.4f}")
        print(f"Difference        : {self.train_accuracy - self.test_accuracy:.4f}")

        print("\nClassification Report")
        print(
            classification_report(
                y_test,
                predictions
            )
        )

        print("\nConfusion Matrix")
        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        return self.test_accuracy, predictions