import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_real_data import load_real_data
from features.feature_extraction import extract_features
from models.svm_model import SVMModel


# Load the dataset
X, y, encoder = load_real_data()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

print("Train labels:", np.bincount(y_train))
print("Test labels :", np.bincount(y_test))

# Extract statistical features from the CSI data
F_train = extract_features(X_train)
F_test = extract_features(X_test)

print("Training feature shape:", F_train.shape)
print("Test feature shape    :", F_test.shape)

# Standardize features before training the SVM
scaler = StandardScaler()

F_train = scaler.fit_transform(F_train)
F_test = scaler.transform(F_test)

print("Scaled training feature shape:", F_train.shape)
print("Scaled test feature shape    :", F_test.shape)

# Train the SVM classifier
model = SVMModel()
model.train(F_train, y_train)

# Evaluate model performance on the test set
accuracy, predictions = model.evaluate(
    F_test,
    y_test
)

print(f"Test Accuracy: {accuracy * 100:.2f}%")