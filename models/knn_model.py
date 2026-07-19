import numpy as np
from sklearn.neighbors import KNeighborsClassifier #sklearns fertige knn implementierung
from sklearn.metrics import accuracy_score # berechnet wie viel prozent der Vorhersagen korrekt waren
import time # messen der Laufzeit, wichtig für Algorithmenvergleich mit anderen modellen

"""
k-NN = k-Nearest Neighbors (k nächste Nachbarn)
Die Idee ist simpel: Wenn ein neues Sample klassifiziert werden soll, 
schaut k-NN welche k Trainingssamples am ähnlichsten sind und nimmt die häufigste Klasse davon.
Beispiel mit k=3:
Neues Sample kommt rein
→ Die 3 ähnlichsten Trainingssamples sind: Klasse 1, Klasse 1, Klasse 2
→ Klasse 1 gewinnt (2 von 3)
→ Vorhersage: Klasse 1
"""

class KNNModel:

    def __init__(self, k=5, metric = 'manhattan' ): # Konstruktor, k=5 ist stnadardwert
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors = k, metric = metric)
        self.training_time = None
        self.inference_time = None # zeit benoetigt um eine vorhersage zu machen



    ##################
    # Training
    ##################

    def train(self, X_train, y_train):
        start = time.time() #gibt aktuelle zeit in sekunden zurucek (timer fuer training starten)
        self.model.fit(X_train, y_train) # trainingsprozess
        self.training_time = time.time() - start
        print(f"K-NN Training abgeschlossen in {self.training_time:.4f} Sekunden")


    ##################
    # Vorhersage
    ##################

    def predict(self, X_test):
        start = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start
        print(f"k-NN Vorhersage abgeschlossen in {self.inference_time:.4f} Sekunden")
        return predictions
    

    ##################
    # Auswertung
    ##################

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) # vergleicht echte labels mit vorhersagen, gibt wert zwischen 0 und 1 zurueck
        print(f"k-NN Accuracy: {accuracy *100:.2f}%")
        
        if X_train is not None:
            train_preds = self.model.predict(X_train)
            train_acc = accuracy_score(y_train, train_preds)
            print(f"k-NN Train Accuracy: {train_acc * 100:.2f}%")

        return accuracy, predictions
    

###################
# Test
###################

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.load_real_data import load_real_data
    from preprocessing.preprocess import preprocess
    from features.feature_extraction import extract_features
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    X, y, encoder = load_real_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
    F_train = extract_features(X_train)
    F_test  = extract_features(X_test)

    print(f"\n{'k':>4} {'Metrik':<12} {'Accuracy':>10}")
    print("-" * 30)
    for metric in ['euclidean', 'manhattan']:
        for k in [1, 3, 5, 7, 10, 15]:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
            clf.fit(F_train, y_train)
            acc = accuracy_score(y_test, clf.predict(F_test))
            print(f"{k:>4} {metric:<12} {acc*100:>9.2f}%")