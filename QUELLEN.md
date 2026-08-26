# Quellen – WiFi Aktivitätserkennung

Gesammelte wissenschaftliche Quellen zur Begründung der Modellwahl im Abschlussbericht.

---

## BiLSTM / BLSTM

**[1] Chen et al. (2019) — Hauptquelle für BiLSTM bei WiFi CSI**
> Chen, Z., Zhang, L., Jiang, C., Cao, Z., & Cui, W. (2019).
> *WiFi CSI Based Passive Human Activity Recognition Using Attention Based BLSTM.*
> IEEE Transactions on Mobile Computing, 18(11), 2714–2724.
> https://www.semanticscholar.org/paper/WiFi-CSI-Based-Passive-Human-Activity-Recognition-Chen-Zhang/0017ece1f39a0ed3a054697e87bc4d92c52a3658

→ Begründung: Validiert BiLSTM direkt für WiFi CSI HAR. Zeigt dass bidirektionale Verarbeitung (vorwärts + rückwärts)
die Erkennungsrate verbessert, weil Aktivitäten zeitliche Muster in beide Richtungen haben. Erreicht ~98% Accuracy.

**[2] Elkelany et al. (2023) — CNN-ABiLSTM für WiFi CSI HAR**
> Elkelany, A., Ross, R., & McKeever, S. (2023).
> *WiFi-Based Human Activity Recognition Using Attention-Based BiLSTM.*
> In: Artificial Intelligence and Cognitive Science. Lecture Notes in Computer Science.
> Springer Nature Switzerland, S. 121–133.
> ISBN: 978-3-031-26438-2. DOI: 10.1007/978-3-031-26438-2_10
> https://link.springer.com/chapter/10.1007/978-3-031-26438-2_10

→ Begründung: Grundlage der Referenzarchitektur (CNN-ABiLSTM). Kombiniert CNN-Feature-Extraktion mit bidirektionalem
LSTM und Attention-Mechanismus. Erreicht 98,54% Accuracy in drei verschiedenen Raumkonfigurationen. Motiviert unsere
Entscheidung, CNN und BiLSTM zu kombinieren.

**[3] ScienceDirect — BiLSTM in Smart Buildings**
> *A Bi-LSTM-based Wi-Fi CSI approach for non-intrusive human behavior recognition in smart buildings.*
> ScienceDirect, Energy and Buildings, 2025.
> https://www.sciencedirect.com/science/article/abs/pii/S0378778825007893

→ Begründung: Bestätigt Anwendbarkeit von BiLSTM für passive, nicht-intrusive Aktivitätserkennung via WiFi CSI in
Innenräumen — identisches Setting wie unser Datensatz.

---

## CNN (1D Convolutional Neural Network)

**[4] Bauer-Wersing (2025) — Grundlagen von CNNs**
> Bauer-Wersing, U. (2025).
> *Computer Vision: Convolution, Filters, Cross-Correlation, Weight Sharing, & Convolutional Neural Networks.*
> Lecture 7, Ringvorlesung Artificial Intelligence.
> Faculty of Computer Science and Engineering, Frankfurt University of Applied Sciences.
> Moodle Course Material.

→ Begründung: Diente als theoretische Grundlage für die Implementierung und Beschreibung der CNN-Architektur. Erläutert
die Funktionsweise von Faltungsoperationen, Filtern, Weight Sharing und Convolutional Neural Networks sowie deren
Anwendung zur automatischen Merkmalsextraktion aus Daten.

**[5] Shang et al. (2021) — LSTM-CNN Hybrid für WiFi CSI HAR**
> Shang, S., Li, X., Zhao, H., Luo, H., & Zhao, Y. (2021).
> *LSTM-CNN Network for Human Activity Recognition Using WiFi CSI Data.*
> Journal of Physics: Conference Series, 1883(1), 012139.
> https://doi.org/10.1088/1742-6588/1883/1/012139

→ Begründung: Belegt direkt die Kombination aus CNN und LSTM für WiFi-CSI-basierte HAR. Zeigt dass tiefere CNN-Stufen (
mehrere Conv-Blöcke) reichhaltigere Feature-Repräsentationen liefern und die Klassifikationsgenauigkeit steigern. Reines
LSTM: 82,72%, LSTM-CNN: 94,14% — motiviert unsere Entscheidung für mehrere CNN-Blöcke.
![img.png](img.png)
**[6] PMC — Critical Analysis of CNN for WiFi CSI HAR**
> *Critical Analysis of Data Leakage in WiFi CSI-Based Human Action Recognition Using CNNs.*
> PMC / NCBI, 2024.
> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11124867/

→ Begründung: Peer-reviewed Analyse von CNN-Ansätzen speziell für WiFi CSI. Zeigt Stärken und typische Fallstricke bei
CNN-Modellen für diesen Anwendungsfall.

**[7] PMC — 1D CNN für HAR (Sensors)**
> *Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening.*
> PMC / NCBI Sensors.
> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5949027/

→ Begründung: Zeigt dass 1D CNNs (Filter über Zeitdimension) effektiv lokale Bewegungsmuster in Zeitreihensignalen
erkennen — Grundlage für unsere 1D-CNN-Implementierung.

**[8] Nature Scientific Reports — DeepConvLSTM**
> *Efficient human activity recognition on edge devices using DeepConv LSTM architectures.*
> Scientific Reports (Nature), 2025.
> https://www.nature.com/articles/s41598-025-98571-2

→ Begründung: Kombination CNN + LSTM bestätigt Synergie beider Architekturen (wie in der Referenzarbeit CNN-ABiLSTM).
CNN extrahiert lokale Features, LSTM verarbeitet zeitliche Abhängigkeiten.

## GRU (Gated Recurrent Unit)

**[9] Mohsen (2023) — GRU for Human Activity Recognition**
> Mohsen, S. (2023).
> *Recognition of Human Activity Using GRU Deep Learning Algorithm.*
> Multimedia Tools and Applications, 82, 47733–47749.
> https://doi.org/10.1007/s11042-023-15571-y

→ Begründung: Belegt die Eignung von GRU-Netzwerken für Human Activity Recognition. Die Arbeit zeigt, dass GRUs
zeitliche Abhängigkeiten in Sensordaten effizient modellieren können und dabei häufig ähnliche Genauigkeiten wie LSTMs
erreichen, jedoch mit weniger Parametern und geringerem Rechenaufwand. Dies motiviert die Verwendung eines GRU-Modells
als Vergleich zu LSTM-basierten Ansätzen.

**[10] GeeksforGeeks — Grundlagen von GRU-Netzwerken**
> GeeksforGeeks.
> *Gated Recurrent Unit Networks.*
> https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/

→ Begründung: Diente als ergänzende technische Referenz zur Erklärung der GRU-Architektur. Beschreibt den Aufbau von
Update Gate und Reset Gate sowie die Unterschiede zwischen GRU und LSTM.

---
---

*Zuletzt aktualisiert: 15.07.2026*
