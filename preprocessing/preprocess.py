import numpy as np
from sklearn.model_selection import train_test_split #Teilt daten automaitsch in training und test auf
from sklearn.preprocessing import StandardScaler #normalisiert daten

"""
Vorbereitung der Daten. Es Passieren 2 Dinge:
-> Normalisierung: alle Werte auf denselben Wertebereich bringen.
                   ML-Modelle arbeiten besser wenn alle Features ähnlich groß sind. 
                   Ohne Normalisierung könnte ein Feature mit großen Werten die anderen dominieren.
-> Train/Test-Split: wir teilen die Daten auf: 80% zum Trainieren, 20% zum Testen. 
                     Das Modell lernt auf den Trainingsdaten und wird auf den Testdaten bewertet (Daten die es noch nie gesehen hat). 
                     So messen wir ob das Modell wirklich gelernt hat oder nur auswendig gelernt hat.
"""


def preprocess(X, y, test_size=0.2): #20% der Daten gehen ins test set, also 80% ins training
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42, # reproduzierbarkeit
        stratify=y # stellt sicher dass alle 3 Klassen gleichmäßig in Training und Test verteilt sind.
    )

    #csv daten auf 2 dimensionen bringen fuer StandardScaler
    X_train_flat = X_train.reshape(-1, 256)
    X_test_flat = X_test.reshape(-1, 256)

    # Normalisierung
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_flat) # berechnet Mittelwert und Standardabweichung nur auf Trainingsdaten und normalisiert sie.
    X_test  = scaler.transform(X_test_flat) # verwendet dieselben Werte vom Training um die Testdaten zu normalisieren

    # wieder auf 3 dimensionen bringen
    n_train = X_train_flat.shape[0] // 500
    n_test  = X_test_flat.shape[0] // 500

    X_train = X_train.reshape(n_train, 500, 256)
    X_test  = X_test.reshape(n_test, 500, 256)
    return X_train, X_test, y_train, y_test, scaler


# ==================
# Test
# ==================

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.synthetic_data import generate_data

    X, y = generate_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"y_train Klassen: {np.unique(y_train, return_counts=True)}")
    print(f"y_test  Klassen: {np.unique(y_test,  return_counts=True)}")