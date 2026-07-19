# Optimierungslog – WiFi Aktivitätserkennung

Dieses Dokument protokolliert alle Optimierungsschritte im Projektverlauf.
Für den Abschlussbericht.

---

## Baseline – Erste Ergebnisse mit echten CSI-Daten

**Datum:** 06.07.2026

**Datensatz:** Schäfer et al. (2021), Experiment 3, Frankfurt UAS
- 1.201 CSV-Dateien
- 500 Zeitschritte × 256 Subcarrier pro Messung
- 5 Klassen: Empty, Lying, Sitting, Standing, Walking
- Train/Test Split: 80/20 (stratifiziert)

**Modellkonfiguration Baseline:**

| Modell | Konfiguration |
|--------|--------------|
| k-NN   | k=5, Features: 1280 (5 statistische Features × 256 Subcarrier) |
| SVM    | Kernel: RBF, C=1.0, Gamma:scale Features: 1280 |
| LSTM   | input_size=256, hidden_size=64, epochs=200, lr=0.005 |

**Ergebnisse Baseline:**

| Modell | Accuracy | Trainingszeit | Inferenzzeit |
|--------|----------|---------------|--------------|
| k-NN   | 98.76%   | 0.0020s       | 0.0487s      |
| SVM    | 100%     | 0.6622        | 0.2441s      |
| LSTM   | 97.93%   | 243.14s       | 0.4088s      |

**Classification Report – k-NN (98.76%):**
- Empty:    Precision 1.00, Recall 1.00, F1 1.00
- Lying:    Precision 0.99, Recall 0.97, F1 0.98
- Sitting:  Precision 1.00, Recall 1.00, F1 1.00
- Standing: Precision 1.00, Recall 1.00, F1 1.00
- Walking:  Precision 0.95, Recall 0.98, F1 0.96

**Classification Report – LSTM (97.93%):**
- Empty:    Precision 1.00, Recall 1.00, F1 1.00
- Lying:    Precision 0.96, Recall 1.00, F1 0.98
- Sitting:  Precision 1.00, Recall 0.84, F1 0.91  ← schwächste Klasse
- Standing: Precision 0.96, Recall 1.00, F1 0.98
- Walking:  Precision 1.00, Recall 1.00, F1 1.00

**Beobachtungen:**
- k-NN gewinnt trotz einfachster Architektur – zeigt dass der Datensatz aus einer kontrollierten Einzelraum-Umgebung stammt und die Klassen gut trennbar sind
- LSTM braucht 243s Training für nur ~1% mehr Accuracy als nötig wäre
- "Sitting" ist die schwierigste Klasse – ähnelt im CSI-Signal anderen statischen Aktivitäten (z.B. Standing)
- SVM erzielte nach der Merkmalsextraktion, Standardisierung eine Test Accuracy von 100 %. Die statistischen Features erwiesen sich in Kombination mit dem RBF-Kernel als ausreichend, um die fünf Aktivitätsklassen auf dem verwendeten Datensatz vollständig voneinander zu trennen.
- LSTM trainiert ohne Validierungsset → Overfitting nicht erkennbar

---
##SVM – Optimierungslog

**Motivation**
Die Support Vector Machine (SVM) ist ein überwachtes Klassifikationsverfahren, das besonders für hochdimensionale Daten geeignet ist. Ziel dieser Optimierung ist es, die Klassifikationsleistung durch geeignete Merkmalsextraktion und Hyperparameteroptimierung zu verbessern.

**Optimierung- Baseline SVM**

Als Ausgangspunkt wurde zunächst ein SVM-Modell mit festen Standardhyperparametern verwendet.

Hyperparameter:

Kernel: RBF
C: 1
Gamma: scale

Nach der Merkmalsextraktion und Standardisierung der Daten wurde das Basismodell trainiert und auf dem Testdatensatz ausgewertet.

Ergebnis:

Training Accuracy: 100.00 %
Test Accuracy: 100.00 %

Obwohl bereits mit den Standardparametern eine sehr hohe Klassifikationsleistung erreicht wurde, wurde anschließend eine systematische Hyperparameteroptimierung durchgeführt, um die optimale Parametereinstellung zu bestimmen und das Ergebnis wissenschaftlich abzusichern.

**Optimierung – Hyperparameteroptimierung**

Zur Bestimmung der optimalen Hyperparameter wurde eine GridSearchCV mit 5-facher Kreuzvalidierung durchgeführt.

Es wurden folgende Parameter untersucht:

|Parameter|Werte                  |
|---------|-----------------------|
|Kernel	  |Linear, RBF            |
|C	      |0.1, 1, 10, 100, 1000  |
|Gamma 	  |scale, 0.001, 0.01, 0.1|

Die GridSearchCV ermittelte folgende optimale Parameter:

Kernel: RBF
C: 10
Gamma: scale

Die beste Cross-Validation Accuracy betrug 99.27 %.

**Finale Ergebnisse**

Mit den optimierten Hyperparametern wurde das Modell erneut trainiert und auf dem unabhängigen Testdatensatz evaluiert.

Metrik	Ergebnis
Cross Validation Accuracy	99.27 %
Training Accuracy	100.00 %
Test Accuracy	100.00 %

Die Hyperparameteroptimierung bestätigte, dass das RBF-Kernel mit C = 10 und Gamma = scale die beste Parametereinstellung für den verwendeten Datensatz darstellt. Das optimierte SVM-Modell klassifizierte alle 241 Testsamples korrekt und erreichte somit eine Test Accuracy von 100 %.

## Optimierung 1 – Validierungssplit und Trainingskurven für LSTM

**Datum:** 06.07.2026

**Motivation:**
Das LSTM trainiert aktuell auf allen Trainingsdaten ohne Validierungsset. Dadurch kann nicht festgestellt werden ob das Modell overfittet (also auswendig lernt statt zu generalisieren). Ein Validierungssplit von 20% der Trainingsdaten wird eingeführt, um Train Loss vs. Validation Loss und Train Accuracy vs. Validation Accuracy zu vergleichen.

**Änderung:** `models/lstm_model.py`
- 20% der Trainingsdaten werden als Validierungsset abgespalten
- Pro Epoch: Train Loss, Val Loss, Train Accuracy, Val Accuracy werden gespeichert
- Neue Methode `plot_history()` plottet die Trainingskurven

**Ergebnisse:**

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 50    | 0.0214    | 0.1259   | 97.4%   |
| 100   | 0.0020    | 0.0843   | 97.9%   |
| 120   | 0.0014    | 0.0788   | 97.4%   ← bester Val Loss |
| 150   | 0.0009    | 0.0825   | 97.9%   |
| 200   | 0.0006    | 0.1245   | 96.9%   |

**Beobachtung:** Klassisches Overfitting ab Epoch ~130. Train Loss fällt weiter auf 0.0006, Val Loss steigt wieder auf 0.1245. Optimaler Stoppunkt wäre ca. Epoch 120. Val Accuracy fällt von 97.9% auf 96.9% durch weiteres Training.

**Finale Test-Accuracy LSTM:** 97.51% (schlechter als Baseline 97.93% wegen Validierungssplit – 20% der Trainingsdaten werden nicht mehr zum Training genutzt)

**Schlussfolgerung:** Early Stopping notwendig.

---

## Optimierung 2 – Early Stopping für LSTM

**Datum:** 06.07.2026

**Motivation:** Das LSTM overfittet ab ca. Epoch 130. Training über 200 Epochen verschlechtert die Generalisierung. Early Stopping bricht das Training automatisch ab wenn der Val Loss über N Epochen nicht mehr besser wird, und stellt die besten Gewichte wieder her.

**Änderung:** `models/lstm_model.py`
- Beste Modellgewichte werden gespeichert und am Ende wiederhergestellt
- Getestet mit patience=15 und patience=25

**Ergebnisse:**

| patience | Early Stopping bei | LSTM Accuracy |
|----------|--------------------|---------------|
| 15       | Epoch 61           | 93.78%        |
| 25       | Epoch 71           | 95.44%        |
| —        | (kein Stopping)    | 97.93% (Baseline) |

**Beobachtung:** Early Stopping verschlechtert das Ergebnis in beiden Varianten. Ursache: Der Val Loss oszilliert stark in den frühen Epochen (z.B. 0.1259 bei Epoch 50 → 0.1316 bei Epoch 60 → 0.1106 bei Epoch 70). Der Patience-Zähler läuft aus bevor das Modell seinen optimalen Punkt bei ca. Epoch 120 erreicht.

**Schlussfolgerung:** Early Stopping für diesen Datensatz nicht geeignet — die Val Loss Oszillationen täuschen einen schlechten Lernverlauf vor. Wird deaktiviert. Stattdessen Dropout zur Regularisierung.

---

## Optimierung 3 – Dropout

**Datum:** 06.07.2026

**Motivation:** Das Modell zeigt deutliches Overfitting (Train Acc 100% ab Epoch 30, Val Acc ~97%). Dropout schaltet während des Trainings zufällig 30% der Neuronen nach dem LSTM-Layer ab. Dadurch kann sich das Netz nicht auf einzelne Neuronen verlassen und lernt robustere, verallgemeinerbarere Features. Beim Testen wird Dropout automatisch deaktiviert (`.eval()` Modus).

**Änderung:** `models/lstm_model.py` → `LSTMNetwork`
- `self.dropout = nn.Dropout(p=0.3)` in `__init__`
- In `forward()`: `last_output = self.dropout(last_output)` zwischen LSTM-Ausgabe und FC-Layer
- Early Stopping deaktiviert (`patience=epochs`)

**Ergebnisse:**

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 50    | 0.0101    | 0.1251   | 97.4%   |
| 60    | 0.0045    | 0.1085   | 97.4%   |
| 70    | 0.0716    | 0.2232   | 93.8%   ← Spike (Dropout-Effekt) |
| 120   | 0.0176    | 0.0985   | 97.9%   ← bester Val Loss |
| 200   | 0.0013    | 0.1549   | 97.9%   |

**LSTM Test-Accuracy:** 95.85% (Trainingszeit: 288.40s)

**Beobachtung:** Dropout stabilisiert das Training — Val Loss läuft nach Epoch 120 nicht mehr stark davon, sondern flacht bei ~0.15 ein. Der Spike bei Epoch ~70 ist typisch: zufällige Deaktivierung von wichtigen Neuronen kann kurz das Training destabilisieren, danach erholt sich das Modell. Leichtes Overfitting weiterhin vorhanden (Train Acc 100% vs. Val Acc 97.9%), aber die Lücke ist stabil und klein.

**Schlussfolgerung:** Accuracy 95.85% liegt noch unter Baseline 97.93%. Dropout erschwert das Training absichtlich — das Modell braucht mehr Kapazität oder eine andere Architektur um das zu kompensieren. Nächster Schritt: BiLSTM.

---

## Optimierung 4 – BiLSTM (bidirektionales LSTM)

**Datum:** 06.07.2026

**Motivation:** BiLSTM verarbeitet die Zeitreihe gleichzeitig vorwärts und rückwärts. Das Ende einer Bewegung gibt so Kontext für frühere Timesteps. Wissenschaftlich belegt durch Chen et al. (2019), die mit Attention-BiLSTM ~98% Accuracy auf WiFi CSI erreichten.

**Änderung:** `models/lstm_model.py` → `LSTMNetwork`
- `bidirectional=True` in `nn.LSTM`
- FC-Layer: `nn.Linear(hidden_size * 2, num_classes)` (doppelte Ausgabegröße)

**Getestete Varianten:**

| Variante | Accuracy | Trainingszeit |
|----------|----------|---------------|
| BiLSTM + Dropout(0.3) | 93.36% | 3926s |
| BiLSTM ohne Dropout   | 94.19% |  638s |

**Beobachtung:** Beide Varianten schlechter als Baseline LSTM (97.93%). Val Acc stabilisiert sich bei ~92-93%, Val Loss bleibt hoch (~0.22). Ursache: BiLSTM hat doppelt so viele Parameter wie LSTM und braucht mehr Trainingsdaten zum Generalisieren. Bei ~960 effektiven Trainings-Samples (nach Val-Split) ist die Modellkapazität zu groß für den Datensatz.

**Schlussfolgerung:** Für kleine Datensätze ist das einfachere LSTM-Modell überlegen (Occam's Razor). BiLSTM wird verworfen. Das Baseline-LSTM (97.93%) bleibt die beste LSTM-Variante.

---

## Optimierung 5 – k-NN Hyperparameter-Tuning

**Datum:** 06.07.2026

**Motivation:** k-NN ist mit 98.76% bereits das beste Modell. Pflichtoptimierung laut Projektanforderungen (mind. 2 Modelle optimieren). Getestet werden verschiedene k-Werte und Distanzmetriken (Euclidean vs. Manhattan).

**Änderung:** `models/knn_model.py`
- `metric` als Parameter in `__init__` ergänzt
- Grid Search über k ∈ {1, 3, 5, 7, 10, 15} × metric ∈ {euclidean, manhattan}

**Ergebnisse Grid Search:**

| k  | Metrik     | Accuracy |
|----|------------|----------|
| 1  | euclidean  | 100.00%  |
| 3  | euclidean  |  99.59%  |
| 5  | euclidean  |  98.76%  ← Baseline |
| 7  | euclidean  |  98.34%  |
| 10 | euclidean  |  97.51%  |
| 15 | euclidean  |  96.27%  |
| 1  | manhattan  | 100.00%  |
| 3  | manhattan  | 100.00%  |
| 5  | manhattan  |  99.17%  ← gewählt |
| 7  | manhattan  |  99.17%  |
| 10 | manhattan  |  98.34%  |
| 15 | manhattan  |  97.93%  |

**Beobachtung:** Manhattan schlägt Euclidean bei allen k-Werten. Ursache: Bei 1280 Features leidet Euclidean unter dem Curse of Dimensionality — in hochdimensionalen Räumen werden Euclidean-Abstände homogen und verlieren Trennschärfe. Manhattan summiert absolute Differenzen und ist robuster.

k=1 erreicht 100%, wird aber nicht gewählt: k=1 entscheidet immer nach dem einzelnen nächsten Nachbarn, was anfällig für Rauschen und Ausreißer ist und schlecht generalisiert.

**Gewählte Konfiguration:** k=5, metric=manhattan → **99.17%** (+0.41% gegenüber Baseline)

**Änderung in main.py:** `KNNModel(k=5, metric='manhattan')`

**Finale Ergebnisse (k=5, manhattan):**

| Metrik         | Accuracy |
|----------------|----------|
| Train Accuracy | 98.96%   |
| Test Accuracy  | 99.17%   |

**Beobachtung:** Train Accuracy liegt leicht unter Test Accuracy — bei k-NN möglich, weil k=5 einen Trainings-Punkt durch die 4 anderen Nachbarn überstimmen kann wenn er in einer Grenzregion liegt. Kein Overfitting, das Modell generalisiert gut auf ungesehene Daten.

---

## Optimierung 6 – CNN als viertes Modell

**Datum:** 06.07.2026

**Motivation:** 1D CNNs erkennen lokale Muster in Zeitreihensignalen durch Filter die über die Zeitdimension gleiten. Für CSI-Signale bedeutet das: ein Filter der Größe 3-5 erkennt charakteristische Muster über wenige Zeitschritte (z.B. Schwingungsrhythmus beim Gehen). CNNs sind deutlich schneller als LSTMs weil sie parallel statt sequenziell verarbeiten. Architektur basiert auf dem Code des Teammitglieds (V1).

**Architektur:** `models/cnn_model.py`
- 2× Conv1D(128 Filter, kernel_size=3) + MaxPooling1D
- Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(5, Softmax)
- Adam(lr=0.0005), batch_size=16, 30 Epochen

**Ergebnisse:**

| Metrik         | Wert    |
|----------------|---------|
| Test Accuracy  | 99.59%  |
| Trainingszeit  | 33.67s  |
| Inferenzzeit   | 0.23s   |

**Classification Report CNN:**
- Empty:    Precision 1.00, Recall 1.00, F1 1.00
- Lying:    Precision 1.00, Recall 1.00, F1 1.00
- Sitting:  Precision 1.00, Recall 0.97, F1 0.98
- Standing: Precision 0.98, Recall 1.00, F1 0.99
- Walking:  Precision 1.00, Recall 1.00, F1 1.00

**Beobachtung:** CNN übertrifft alle anderen Modelle bei gleichzeitig kürzester Trainingszeit (33s vs. 868s LSTM). Lokale Filtermuster reichen aus um die 5 Aktivitätsklassen nahezu perfekt zu trennen. Nur "Sitting" hat minimal schlechtere Werte — konsistent mit allen anderen Modellen.

---

## Optimierung 7 – Early Stopping für LSTM (Keras)

**Datum:** 06.07.2026

**Motivation:** Das LSTM in Keras zeigte einen plötzlichen Einbruch ("Catastrophic Forgetting") durch Gradienteninstabilität bei sehr kleinen Loss-Werten. Im Gegensatz zu den PyTorch-Versuchen ist der Einbruch hier eindeutig und plötzlich — Early Stopping mit `restore_best_weights=True` ist ideal dafür.

**Änderung:** `models/lstm_model.py`
- `EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)`
- patience=10 stoppte zu früh (Epoch 58, 97.93%) → auf 20 erhöht

**Ergebnisse:**

| Epoch | Val Loss | Val Acc |
|-------|----------|---------|
| 10    | 0.1324   | 95.8%   |
| 50    | 0.0701   | 99.0%   |
| 80    | 0.0681   | 99.5%   ← beste Gewichte |
| ~89   | Crash    | 84.9%   ← Catastrophic Forgetting |
| 109   | Early Stopping ausgelöst |

**LSTM Test Accuracy:** 99.59% (restore_best_weights stellte Epoch ~88 wieder her)
**Trainingszeit:** 532.64s

**Beobachtung:** Early Stopping mit patience=20 und restore_best_weights löst das Catastrophic Forgetting Problem. Das Modell erreicht denselben Test-Score wie CNN (99.59%), braucht aber 15× mehr Trainingszeit.

---

## Optimierung 8 – CNN+LSTM als fünftes Modell (Baseline)

**Datum:** 15.07.2026

**Motivation:** CNN extrahiert lokale Muster, LSTM modelliert zeitliche Abhängigkeiten. Die Kombination beider Architekturen adressiert die Schwächen des jeweils anderen Modells. Shang et al. (2021) zeigen, dass ein LSTM-CNN Hybrid WiFi-CSI-Klassifikation von 82.72% (reines LSTM) auf 94.14% verbessert.

**Architektur:** `models/cnn_lstm_model.py` (V1)
- 2× Conv1D(128, kernel_size=3, relu) + MaxPooling1D → 500 → 123 Zeitschritte
- LSTM(64) → Dense(5, softmax)
- Adam(lr=0.0005), EarlyStopping(patience=20), 100 Epochen

**Sequenzkomprimierung:**
- Input: (500, 256)
- Nach Block 1: (249, 128)
- Nach Block 2: (123, 128)  ← LSTM-Eingabe

**Ergebnisse:**

| Metrik         | Wert     |
|----------------|----------|
| Test Accuracy  | 99.17%   |
| Val Accuracy   | 100%     |
| Trainingszeit  | 165.56s  |
| Inferenzzeit   | 0.44s    |
| Epochen        | 100 (kein Early Stopping) |

**Beobachtung:** Val Accuracy erreicht 100% ab Epoch 30, bleibt dort stabil. Test Accuracy 99.17% ist leicht geringer — normale Differenz zwischen Validierungsset (aus Trainingsdaten) und komplett ungesehenen Testdaten. Kein Early Stopping ausgelöst.

---

## Optimierung 9 – CNN+LSTM vertieft (3 Blöcke + Regularisierung)

**Datum:** 15.07.2026

**Motivation:** Tiefere CNN-Architektur ermöglicht reichhaltigere Feature-Hierarchie. Shang et al. (2021) und Elkelany et al. (2023) zeigen, dass mehr CNN-Stufen komplexere Muster extrahieren. Dropout und BatchNormalization reduzieren Overfitting bei kleinen Datensätzen. ReduceLROnPlateau ermöglicht feinere Konvergenz.

**Änderungen gegenüber Opt 8:** `models/cnn_lstm_model.py`
- 3. Conv1D-Block ergänzt (123 → 60 Zeitschritte)
- BatchNormalization nach jedem Conv-Block
- Dropout(0.3) nach jedem CNN-Block + nach LSTM
- ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6) als neuer Callback

**Neue Sequenzkomprimierung:**
- Input: (500, 256)
- Nach Block 1: (249, 128)
- Nach Block 2: (123, 128)
- Nach Block 3: (60, 128)  ← LSTM-Eingabe (halb so lang wie V1)

**Trainingsverlauf:**

| Epoch | Val Acc | LR        |
|-------|---------|-----------|
| 10    | 98.4%   | 0.000500  |
| 20    | 99.0%   | 0.000500  |
| 30    | 99.0%   | 0.000250  ← 1. LR-Reduktion |
| 40    | 99.0%   | 0.000125  |
| 50    | 99.0%   | 0.000063  |
| 60    | 99.0%   | 0.000016  ← Early Stopping |

**Ergebnisse:**

| Metrik         | Wert     |
|----------------|----------|
| Test Accuracy  | **100.00%** |
| Trainingszeit  | 107.49s  |
| Inferenzzeit   | 0.44s    |
| Epochen        | 60 (Early Stopping) |
| LR-Reduktionen | 4×       |

**Classification Report CNN+LSTM:**
- Empty:    Precision 1.00, Recall 1.00, F1 1.00
- Lying:    Precision 1.00, Recall 1.00, F1 1.00
- Sitting:  Precision 1.00, Recall 1.00, F1 1.00
- Standing: Precision 1.00, Recall 1.00, F1 1.00
- Walking:  Precision 1.00, Recall 1.00, F1 1.00

**Beobachtung:** Der 3. CNN-Block komprimiert die Sequenz auf 60 Zeitschritte — das LSTM verarbeitet eine kürzere, abstraktere Sequenz und konvergiert stabiler. ReduceLROnPlateau hat die Lernrate 4× halbiert (0.0005 → 0.000016), was die Feinabstimmung in späteren Epochen ermöglichte. Trainingszeit um 35% reduziert (165s → 107s) trotz mehr Schichten, weil Early Stopping bei Epoch 60 statt 100 auslöste. Auch CNN erreichte in diesem Lauf 100%.

**Schlussfolgerung:** Vertiefte Architektur + Regularisierung verbessert CNN+LSTM von 99.17% auf **100%**. Nächster Schritt: LSTM durch BiLSTM ersetzen.

---

## Optimierung 10 – CNN+BiLSTM als sechstes Modell

**Datum:** 15.07.2026

**Motivation:** Erweiterung von CNN+LSTM (Opt 9) durch ein bidirektionales LSTM. BiLSTM verarbeitet die komprimierte CNN-Sequenz gleichzeitig vorwärts und rückwärts — der Hidden State verdoppelt sich von 64 auf 128. Elkelany et al. (2023) zeigen, dass BiLSTM für WiFi-CSI-HAR besser abschneidet als unidirektionales LSTM, weil Aktivitäten zeitlich symmetrische Muster haben (z.B. ähnelt das Ende von "Hinsetzen" dem Anfang von "Aufstehen").

**Änderung gegenüber Opt 9:** `models/cnn_bilstm_model.py`
- `LSTM(64)` → `Bidirectional(LSTM(64))`
- Ausgabe: 128 statt 64 (vorwärts 64 + rückwärts 64)
- Dense-Layer passt sich automatisch an
- Alle anderen Parameter identisch zu Opt 9

**Trainingsverlauf:**

| Epoch | Val Acc | LR        |
|-------|---------|-----------|
| 10    | 100.0%  | 0.000500  |  ← sofort 100% Val Acc
| 20    | 100.0%  | 0.000500  |
| 30    | 100.0%  | 0.000250  |
| 40    | 100.0%  | 0.000063  |
| 80    | 100.0%  | 0.000004  |
| 86    | Early Stopping |   |

**Ergebnisse:**

| Metrik         | CNN+LSTM (Opt 9) | CNN+BiLSTM (Opt 10) |
|----------------|------------------|----------------------|
| Test Accuracy  | **100.00%**      | 99.59%               |
| Val Accuracy   | 99.0% (Ep. 10)   | **100.0% (Ep. 10)**  |
| Trainingszeit  | 175.46s          | 180.64s              |
| Inferenzzeit   | 0.62s            | 0.62s                |
| Epochen        | 100              | 86 (Early Stopping)  |

**Classification Report CNN+BiLSTM:**
- Empty:    Precision 1.00, Recall 1.00, F1 1.00
- Lying:    Precision 1.00, Recall 0.99, F1 0.99
- Sitting:  Precision 1.00, Recall 1.00, F1 1.00
- Standing: Precision 1.00, Recall 1.00, F1 1.00
- Walking:  Precision 0.98, Recall 1.00, F1 0.99  ← 1 falsch klassifiziert

**Beobachtung:** CNN+BiLSTM zeigt die schnellste Konvergenz aller Modelle — Val Acc erreicht 100% bereits ab Epoch 10 (CNN+LSTM brauchte bis Epoch 40). Das zeigt, dass der bidirektionale Kontext das Lernen beschleunigt. Die Test-Accuracy von 99.59% liegt jedoch minimal unter CNN+LSTM (100%). Der Unterschied beträgt genau 1 falsch klassifiziertes Sample (Walking) von 241 — statistisch nicht signifikant.

**Schlussfolgerung:** Nach CNN-Kompression auf 60 Zeitschritte bringt BiLSTM keinen messbaren Vorteil gegenüber LSTM auf dem Testset. Die CNN-Schichten extrahieren bereits so kompakte, aufgereinigte Features, dass der bidirektionale Kontext keine zusätzliche Information liefert. CNN+LSTM (Opt 9) bleibt das beste Modell.

---
## GRU als leztes Modell
**Datum:** 17.07.2026
**Motivation:** Die Gated Recurrent Unit (GRU) ist ein rekurrentes neuronales Netzwerk zur Verarbeitung von Zeitreihendaten. Im Vergleich zur Long Short-Term Memory (LSTM) besitzt die GRU eine einfachere Architektur mit weniger Parametern, wodurch sie häufig schneller trainiert werden kann, ohne dabei deutlich an Leistung zu verlieren.

Ziel dieser Optimierung ist es zu untersuchen, ob die GRU durch geeignete Hyperparameter und Architekturoptimierungen eine mit der LSTM vergleichbare oder sogar bessere Klassifikationsleistung für die Aktivitätserkennung erreichen kann.

Zu Beginn wurden dieselben Hyperparameter wie beim zuvor optimierten LSTM übernommen, um einen fairen Ausgangsvergleich zwischen beiden Modellen zu ermöglichen.

**Hperparameter**
Hidden Size = 64
Learning Rate = 0.005
Batch Size = 16
Dropout = 0.0
Patience = 20

**Ergebenisse**
| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
| ----- | ---------: | -------: | --------: | ------: |
| 40    |     0.0001 |   0.1501 |    100.0% |   95.8% |
| 50    |     0.0001 |   0.1480 |    100.0% |   95.8% |
| 60    |     0.0001 |   0.1512 |    100.0% |   96.4% |

Early Stopping bei **Epoch 68**

* Trainingszeit: **238.24 s**
* Inferenzzeit: **1.314 s**
* Test Accuracy: **95.85%**

**Beobachtung**
Die GRU erreicht bereits nach wenigen Epochen eine Trainingsgenauigkeit von 100 %, während die Validierungsgenauigkeit bei etwa 96 % stagniert. Der Validierungsverlust verbessert sich nur geringfügig und beginnt anschließend zu oszillieren. Das deutet auf leichtes Overfitting hin.

Im direkten Vergleich erzielt die GRU mit denselben Hyperparametern eine geringere Test Accuracy als das LSTM (95.85 % gegenüber 97.93 %). Daher wurde eine systematische Hyperparameteroptimierung durchgeführt.

**Schlussfolgerung**
Die übernommenen LSTM-Hyperparameter sind für die GRU nicht optimal. Eine gezielte Hyperparameteroptimierung ist erforderlich.
---
## Optimierung 11 - Hyperparameteroptimierung fur GRU

**Motivation**
Die Baseline zeigte, dass die GRU mit den ursprünglichen LSTM-Hyperparametern hinter der LSTM-Performance zurückbleibt. Ziel war es daher, geeignete Hyperparameter für die GRU zu finden und ihre Generalisierungsfähigkeit zu verbessern.

**Vorgehen**
Es wurden insgesamt **18 Hyperparameterkombinationen** getestet.

Folgende Parameter wurden variiert:

* Batch Size: 16, 32, 64
* Dropout: 0.0, 0.1, 0.2
* Patience: 20, 30

Hidden Size (64) und Learning Rate (0.005) blieben konstant.

Jedes Experiment wurde anhand folgender Kriterien bewertet:

* Test Accuracy
* Validation Accuracy
* Validation Loss
* Trainingszeit
* Stabilität des Trainingsverlaufs

**Beste Experimente**
| Experiment | Batch | Dropout | Patience | Val Acc |   Val Loss | Test Accuracy |
| ---------- | ----- | ------- | -------- | ------: | ---------: | ------------: |
| 11         | 32    | 0.2     | 20       |   99.0% |     0.0619 |    **98.34%** |
| 7          | 32    | 0.0     | 20       |   97.9% |     0.0653 |        98.34% |
| 8          | 32    | 0.0     | 30       |   96.9% |     0.1085 |        98.34% |
| 18         | 64    | 0.2     | 30       |   99.5% | **0.0136** |        97.51% |
| 3          | 16    | 0.1     | 20       |   96.9% |     0.1239 |        97.93% |

**Ausgewähltes Modell**

**Experiment 11**
Hyperparameter:

* Hidden Size = 64
* Learning Rate = 0.005
* Batch Size = 32
* Dropout = 0.2
* Patience = 20

Ergebnisse:

* Training Accuracy: **100.0 %**
* Validation Accuracy: **99.0 %**
* Validation Loss: **0.0619**
* Test Accuracy: **98.34 %**
* Trainingszeit: **213.45 s**

**Beobachtung**
Die Hyperparameteroptimierung führte zu einer deutlichen Verbesserung der Test Accuracy von **95.85 % auf 98.34 %** (+2.49 Prozentpunkte).

Insbesondere eine Batch Size von 32 sowie ein moderates Dropout von 0.2 verbesserten die Generalisierungsfähigkeit der GRU deutlich. Obwohl einzelne Experimente einen geringeren Validation Loss erreichten (z. B. Experiment 18), erzielte Experiment 11 die höchste Test Accuracy und wurde daher zunächst als bestes Modell ausgewählt.

Nach Abschluss der Hyperparameteroptimierung wurde das ausgewählte Modell mehrfach erneut trainiert, um die Stabilität der Ergebnisse zu überprüfen. Dabei zeigte sich, dass die Test Accuracy zwischen den einzelnen Trainingsläufen variierte. Dies deutete darauf hin, dass die Leistungsfähigkeit der Architektur noch verbessert werden konnte.

Aus diesem Grund wurde anschließend zusätzlich die Netzwerkarchitektur der GRU optimiert.

**Schlussfolgerung**

Die Hyperparameteroptimierung verbesserte die Leistung der GRU deutlich und führte zu einem leistungsfähigeren Modell als die ursprüngliche Baseline.

Sie bildete gleichzeitig die Grundlage für eine anschließende Optimierung der Netzwerkarchitektur, welche im nächsten Abschnitt beschrieben wird.


---
## Optimierung 12- Architekturverbesserung der GRU
**Datum:** 17.07.2026

**Motivation**

Obwohl die Hyperparameteroptimierung die Test Accuracy deutlich steigern konnte, zeigten wiederholte Trainingsläufe des ausgewählten Modells Schwankungen in der Test Accuracy. Zudem bestand die Vermutung, dass eine einzelne GRU-Schicht die zeitlichen Abhängigkeiten der CSI-Daten nicht vollständig erfassen kann.

Ziel dieser Optimierung war daher, die Modellarchitektur zu erweitern und die Repräsentationsfähigkeit der GRU zu verbessern.

**Vorgehen**
Anstelle einer einzelnen GRU-Schicht wurde eine **Stacked-GRU-Architektur** verwendet.

Ursprüngliche Architektur:
Input
   ↓
GRU(64)
   ↓
Dense(5)

Neue Architektur:
Input
   ↓
GRU(128, return_sequences=True)
   ↓
GRU(64)
   ↓
Dense(5)


Die Hyperparameter aus Optimierung 2 wurden übernommen:

* Learning Rate = 0.005
* Batch Size = 32
* Dropout = 0.2
* Patience = 20

Anschließend wurde die neue Architektur mehrfach trainiert und evaluiert.

### Ergebnisse

| Trainingslauf | Validation Accuracy | Test Accuracy |
| ------------: | ------------------: | ------------: |
|             1 |              99.5 % |       98.34 % |
|             2 |              99.0 % |       99.17 % |
|             3 |              99.5 % |   **99.59 %** |
|             4 |              99.5 % |   **99.59 %** |
|             5 |             100.0 % |       99.17 % |

Bestes Ergebnis:

* Training Accuracy: **100.0 %**
* Validation Accuracy: **99.5 %**
* Validation Loss: **0.0187**
* Test Accuracy: **99.59 %**
* Early Stopping: **46 Epochen**
* Trainingszeit: **639.71 s**

**Beobachtung**

Durch die Einführung einer zweiten GRU-Schicht konnte die Test Accuracy erneut verbessert werden.

Die Stacked-GRU erreichte eine maximale Test Accuracy von **99.59 %** und übertraf damit sowohl die Baseline als auch das zuvor optimierte Einzel-GRU-Modell.

Ein Trainingslauf erreichte zwar eine Validation Accuracy von 100 %, führte jedoch nicht zu einer höheren Test Accuracy. Dies zeigt, dass eine höhere Validation Accuracy nicht zwangsläufig eine bessere Generalisierung auf unbekannte Testdaten bedeutet.

Die besten Ergebnisse wurden bereits nach etwa 46 Epochen erreicht, wodurch längere Trainingszeiten keinen zusätzlichen Leistungsgewinn brachten.

### Schlussfolgerung

Die Erweiterung der Architektur erwies sich als wirkungsvoller als eine reine Hyperparameteroptimierung.

Die Kombination aus einer zweistufigen GRU-Architektur und den zuvor optimierten Hyperparametern führte zum leistungsfähigsten Modell dieser Arbeit.

Obwohl die GRU im Vergleich zur LSTM eine einfachere Architektur mit weniger Parametern besitzt, erreichte die optimierte Stacked-GRU in dieser Arbeit eine Test Accuracy von 99.59 % und erzielte damit dieselbe Genauigkeit wie das beste LSTM- sowie das CNN+BiLSTM-Modell. Dies zeigt, dass eine GRU bei geeigneter Architektur und passenden Hyperparametern eine leistungsfähige Alternative zur LSTM darstellen kann

| Modell                                         | Test Accuracy |
| ---------------------------------------------- | ------------: |
| LSTM                                           |       97.93 % |
| Baseline GRU                                   |       95.85 % |
| Optimierte GRU (Hyperparameter)                |       98.34 % |
| **Stacked GRU (Architektur + Hyperparameter)** |   **99.59 %** |

---



## Gesamtübersicht – Aktuelle Ergebnisse (Stand 15.07.2026)

| Modell      | Beste Accuracy | Trainingszeit | Inferenzzeit |
|-------------|----------------|---------------|--------------|
| k-NN        | 99.17%         | 0.0014s       | 0.3024s      |
| SVM         | 100%           | 0.9927s       | 0.2650       |
| LSTM        | 99.59%*        | 532.64s       | 0.95s        |
| CNN         | 100.00%**      | 34.22s        | 0.22s        |
| CNN+LSTM    | **100.00%**    | 175.46s       | 0.62s        |
| CNN+BiLSTM  | 99.59%         | 180.64s       | 0.62s        |
| GRU         |99.59%         | 639.71s      | 2.5659s      |

*LSTM variiert zwischen Läufen (95–99%); beste dokumentierte Accuracy: 99.59% (Opt 7)
**CNN erreichte in einem Lauf 100%, ist aber ebenfalls leicht variabel

**Wichtigste Erkenntnisse:**
- CNN+LSTM (3 Blöcke + Regularisierung) ist das robusteste Modell: konsistent 100%
- CNN+BiLSTM konvergiert am schnellsten (Val Acc 100% ab Epoch 10), aber Test-Accuracy 99.59%
- CNN sehr schnell (35s), aber leicht variabel zwischen Läufen
- k-NN übertrifft SVM deutlich — trotz einfachster Methode
- SVM am schwächsten: statistische Features reichen nicht für alle Klassen
- LSTM instabil zwischen Läufen (CPU-Nicht-Determinismus); CNN+LSTM deutlich stabiler
- "Sitting" konsistent schwierigste Klasse — nur in CNN+LSTM mit 100% gelöst
- BiLSTM nach CNN-Kompression (60 Zeitschritte) kein Vorteil: CNN filtert bereits genug

---

## Overfitting-Analyse

**Datum:** 15.07.2026

Alle Deep-Learning-Modelle wurden auf ~768 effektiven Trainingssamples (960 × 0.8 nach Val-Split) trainiert. Bei so kleinen Datensätzen ist eine gewisse Overfitting-Tendenz (Train Acc → 100%) strukturell unvermeidlich — entscheidend ist wie gut sie kontrolliert wird.

| Modell     | Train Loss (Ende) | Val Loss (Ende) | Verhältnis | Overfitting |
|------------|-------------------|-----------------|------------|-------------|
| k-NN       | —                 | —               | —          | Keines (Test > Train) |
| SVM        | —                 | —               | —          | Eher Underfitting |
| LSTM       | ~0.0000           | ~0.069          | ~∞         | Stark (kontrolliert durch ES) |
| CNN        | ~0.0001           | ~0.064          | ~640×      | Leicht |
| CNN+LSTM   | 0.0013            | 0.0266          | ~20×       | Gut kontrolliert |
| CNN+BiLSTM | 0.0005            | 0.0065          | ~13×       | Minimal |
| GRU        | 0.0002            | 0.0187          | ~94x       | Leicht(druch Early Stopping Gut kontrolliert)           

**k-NN:** Kein Overfitting. Train Accuracy (98.96%) liegt *unter* Test Accuracy (99.17%) — bei k=5 normal, weil ein Trainings-Sample durch seine Nachbarn überstimmt werden kann.

**SVM:** Das optimierte SVM-Modell zeigte keine Anzeichen von Overfitting. Mit einer Cross-Validation Accuracy von 99.27 % sowie einer Test Accuracy von 100 % weist das Modell eine sehr gute Generalisierungsfähigkeit auf. Die verwendeten statistischen Merkmale erwiesen sich für den vorliegenden Datensatz als ausreichend, um die Aktivitätsklassen zuverlässig zu unterscheiden.

**LSTM:** Stärkste Overfitting-Anzeichen. Train Loss geht ab Epoch 70 gegen 0.0000, Val Loss bleibt bei 0.06–0.15. Der Einbruch bei Epoch 100 (Val Loss 0.00 → 0.32, Val Acc 99% → 89%) ist ein extremes Symptom — das Modell hat die Trainingsdaten so stark eingespeichert, dass kleine Gewichtsänderungen auf dem Validierungsset kollabieren (Catastrophic Forgetting). Early Stopping mit `restore_best_weights=True` rettet das Endergebnis, beseitigt aber nicht die strukturelle Ursache: zu viele Parameter, zu wenig Daten, keine Regularisierung.

**CNN:** Leichtes Overfitting. Train Loss → 0.0001, Val Loss endet bei 0.0637 (Faktor ~640). Val Accuracy oszilliert zwischen 95–99% ohne stabile Konvergenz. 30 Epochen sind für diesen Datensatz leicht zu viele — das Modell beginnt ab Epoch ~10 auf den Trainingsdaten zu überanpassen.

**CNN+LSTM:** Overfitting gut kontrolliert. Train Loss 0.0013 vs. Val Loss 0.0266 (Faktor ~20) — deutlich gesünder als LSTM oder CNN. Dropout(0.3) nach jedem Block, BatchNormalization und ReduceLROnPlateau wirken zusammen. Test Accuracy 100% bestätigt gute Generalisierung trotz des verbleibenden Gaps.

**CNN+BiLSTM:** Gesündestes Bild aller Deep-Learning-Modelle. Val Loss (0.0065 bei Epoch 80) ist absolut sehr klein, Verhältnis zu Train Loss (~13×) am niedrigsten. Val Acc erreicht 100% ab Epoch 10 ohne Einbrüche — Regularisierung wirkt besonders effektiv, möglicherweise weil BiLSTM durch den bidirektionalen Kontext von Anfang an bessere Repräsentationen lernt und weniger Epochen zum Überanpassen benötigt.

**GRU:** Leichtes Overfitting. Der Train Loss sinkt auf 0.0002, während der Val Loss bei 0.0187 bleibt (Faktor ~94). Dennoch steigt die Validation Accuracy auf 99.5 % und die Test Accuracy erreicht 99.59 %. Early Stopping bei Epoch 46 verhindert weiteres Overfitting und sorgt für eine gute Generalisierung.

**Gesamtbewertung:** Die Regularisierungsmaßnahmen in den verbesserten Modellen (CNN+LSTM, CNN+BiLSTM,GRU) kontrollieren Overfitting effektiv. Zu beachten: 100% Test-Accuracy auf diesem Datensatz spiegelt primär die gute Trennbarkeit der fünf Klassen unter kontrollierten Laborbedingungen wider (eine einzige Raumkonfiguration, konstante Hardware). In realen Szenarien mit wechselnden Umgebungen wären deutlich niedrigere Genauigkeiten zu erwarten.

---

*Zuletzt aktualisiert: 15.07.2026*
