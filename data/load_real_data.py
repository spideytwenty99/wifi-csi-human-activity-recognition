import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data/csv'

def load_real_data():
    X = [] # Liste fuer tatsaechliche csv daten (signal)
    y = [] # Liste fuer label (name der Klasse)
    
    for file in sorted(os.listdir(DATA_PATH)):
        if file.endswith('.csv'):
            label = file.split('_')[0] #teilt den Dateinamen am _ auf und nimmt das erste Teil, also nur das label oder die Klasse (zb Lying)
            
            filepath = os.path.join(DATA_PATH, file)
            df = pd.read_csv(filepath) #liest csv file ein

            X.append(df) # Signaldaten zur Liste hinzufuegen
            y.append(label) # label zur Liste hinzufuegen
            
           # print(f"{file} → Label: {label}, Shape: {df.shape}") # zeigt dir wie viele Zeilen und Spalten die Datei hat

    # Listen in arrays umwalndeln
    X = np.array(X)
    y = np.array(y)

    print("\n========== Dataset Summary ==========")
    print(f"Total Files Loaded : {len(X)}")
    print(f"Each File Shape    : {X[0].shape}")
    print(f"Dataset Shape      : {X.shape}")
    print(f"Classes            : {np.unique(y)}")
    print("=====================================")

    # LabelEncoder der die Strings in Zahlen umwandelt
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print(f"Labels als Zahlen: {np.unique(y)}")
    print(f"Bedeutung: {encoder.classes_}")

    return X, y, encoder # zurückgeben, weil wir ihn später noch brauchen um Zahlen wieder in Klassennamen umzuwandeln fuer plots


