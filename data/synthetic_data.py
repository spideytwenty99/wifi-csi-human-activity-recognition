import numpy as np

#constants
N_SAMPLES = 300 #100 messungen pro aktivitaet
TIMESTEPS = 50 #jede Messung hat 50 Zeitschritte (50 aufeinanderfolgende CSI Werte)

# =======================
# Datengenerierung
# =======================

#Wir generieren 3 klassen von synthetischen CSI Daten: Keine Bewegung, Gehen und Winken


def generate_class0(n, t):
    """
     Klasse 0: Keine Bewegung
     → Signal bleibt stabil, nur leichtes Rauschen
     n = wie viele samples, t = wie viele zeitschritte
    """
    data = [] # leere Liste, hier werden die Signale gesammelt
    for _ in range(n): #generiert n samples (signnal) bestehend aus Mittelwert + noise
        noise = np.random.normal(0, 0.1, t) #kleines Rauschen
        signal = 1.0 + noise #stabiler Mkttelwert bei 1.0
        data.append(signal)
    return np.array(data)


def generate_class1(n, t):
    """
    Klasse 1: Gehen
    → Regelmäßige, rhythmische Sinusschwingung (Schrittfrequenz)
    """
    data = []
    for _ in range(n):
        time = np.linspace(0, 2 * np.pi, t) # erzeugt t gleichmäßig verteilte Zeitpunkte von 0 bis 2π (Sinusperiode)
        freq = np.random.uniform(0.8, 1.2)     # zufällige Frequenz zwischen 0.8 und 1.2. Das simuliert natürliche Variation (nicht jeder geht glecih schnell).
        noise = np.random.normal(0, 0.15, t)
        signal = np.sin(freq * time) + noise #sinuswelle + rauschen (gehen hat rythmischen takt)
        data.append(signal)
    return np.array(data)


def generate_class2(n, t):
    """
    Klasse 2: Winken
    → Kurze schnelle Ausschläge (höhere Frequenz, größere Amplitude)
    """
    data = []
    for _ in range(n):
        time = np.linspace(0, 4 * np.pi, t) # diesmal zwei volle Perioden (4π statt 2π), weil Winken schneller ist.
        freq = np.random.uniform(2.5, 3.5)     # höhere Frequenz als beim gehen
        amp  = np.random.uniform(1.5, 2.5)     # größere Amplitude (Winken erzeugt hoehere Signalausschläge)
        noise = np.random.normal(0, 0.2, t)
        signal = amp * np.sin(freq * time) + noise # amplitude multipliziert die hoehe der Welle
        data.append(signal)
    return np.array(data)


# =======================================
# Hauptfunktion - 3 Datenklassen erzeugen
# =======================================

def generate_data():
    np.random.seed(42) #reproduzierbarkeit
    
    X0 = generate_class0(N_SAMPLES, TIMESTEPS)
    X1 = generate_class1(N_SAMPLES, TIMESTEPS)
    X2 = generate_class2(N_SAMPLES, TIMESTEPS)
    
    X = np.vstack([X0, X1, X2]) #stapelt 3 arrays untereinander, shape (300 samples, 50 zeitschritte)
    y = np.array([0]*N_SAMPLES + [1]*N_SAMPLES + [2]*N_SAMPLES) # erzeugt die Labels: 100x die 0, 100x die 1, 100x die 2.
    
    return X, y


# =======================
# Test
# =======================
if __name__ == "__main__":
    X, y = generate_data()
    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    print(f"Klassen: {np.unique(y)}")