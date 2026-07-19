import numpy as np

"""
nach dem preprocessing haben unsere Daten die Form (300, 50) — also 50 rohe Zeitschritte pro Sample. 
Das Problem: SVM und k-NN können nicht gut mit rohen Zeitreihen umgehen. 
Die brauchen einzelne aussagekräftige Zahlen pro Sample — sogenannte Features.
Wir extrahieren aus jedem Signal 5 statistische Features:
    Mittelwert   → wie hoch ist das Signal im Durchschnitt?
    Varianz      → wie stark schwankt es?
    Maximum      → größter Ausschlag
    Minimum      → kleinster Ausschlag
    Energie      → Summe der quadrierten Werte (Gesamtstärke)

    Ein Sample mit 50 Zeitschritten wird also zu einem Vektor mit 5 Zahlen. Das nennt man Feature Vector.
"""

def extract_features(X):
    """
    Extrahiert statistische Features aus jeder Zeitreihe.
    
    Input:  X shape (n_samples, 500, 256)
    Output: F shape (n_samples, 1280) - jedes sample hat je 1280 features
    """
    features = [] # Behälter für alle samples
    

    for sample in X: # geht durch jedes der 300 zeitreihen einzeln
        sample_features = [] # temporärer Behälter für ein Sample
        for i in range(256): # durvch alle subcarrier des jeweiligen samples iterieren
            subcarrier = sample[:, i]
            mittelwert = np.mean(subcarrier) # durchschnitt aller 256 werte
            varianz    = np.var(subcarrier) # wie stark streuen werte? bei keiner Bewegung klein, bei Winken bspw groß
            maximum    = np.max(subcarrier)
            minimum    = np.min(subcarrier)
            energie    = np.sum(subcarrier ** 2) # Signalenergie

            sample_features.extend([mittelwert, varianz, maximum, minimum, energie])

        features.append(sample_features) # fuegt alles dem feature vektor hinzu

    return np.array(features)


##############
# Test
##############

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.synthetic_data import generate_data

    X, y = generate_data()
    F = extract_features(X)

    print(f"X Shape vorher: {X.shape}")
    print(f"F Shape nachher: {F.shape}")
    print(f"\nBeispiel Feature Vector (Sample 0): {F[0]}")
    print(f"Bedeutung: [Mittelwert, Varianz, Max, Min, Energie]")