import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.load_real_data import load_real_data
from preprocessing.preprocess import preprocess
from features.feature_extraction import extract_features
from models.svm_model import SVMModel

# Load dataset
X, y, encoder = load_real_data()

# Preprocessing
# Train/Test Split
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


# Feature Extraction
F_train = extract_features(X_train)
F_test = extract_features(X_test)

print(F_train.shape)
print(F_test.shape)
# Feature Scaling (for SVM)
scaler = StandardScaler()

F_train = scaler.fit_transform(F_train)
F_test = scaler.transform(F_test)

print(F_train.shape)
print(F_test.shape)
# Train SVM
model = SVMModel()
model.train(F_train, y_train)

# Evaluate
accuracy, predictions = model.evaluate(F_test, y_test)