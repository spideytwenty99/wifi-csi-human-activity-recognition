# WiFi CSI Human Activity Recognition

Project on **human activity recognition** using **Wi-Fi Channel State Information (CSI)** and machine learning/deep learning models.

---

## 📖 Project Overview

This project investigates the use of Wi-Fi CSI signals for recognizing indoor human activities without requiring wearable sensors or cameras.

Several classical machine learning and deep learning models are implemented and evaluated using the same dataset and preprocessing pipeline.


---
## 📊 Dataset

This project uses the **Experiment 3** dataset from the paper:

> **Schäfer, J., Barrsiwal, B. R., Kokhkharova, M., Adil, H., & Liebehenschel, J. (2021). _Human Activity Recognition Using CSI Information with Nexmon_. Applied Sciences, 11(19), 8860.**

The dataset contains **WiFi Channel State Information (CSI) amplitude measurements** collected in a controlled indoor **Line-of-Sight (LOS)** environment using commodity WiFi devices and the **Nexmon CSI framework**.

For this project, the **Experiment 3** dataset was selected because it provides preprocessed CSI amplitude data in CSV format, making it suitable for evaluating machine learning and deep learning models for Human Activity Recognition (HAR).

### Activities

The dataset consists of five activity classes:

- Empty
- Lying
- Sitting
- Standing
- Walking

Each CSV file represents one recorded WiFi CSI sample and contains:

- **500 CSI packets**
- **256 subcarriers**
- CSI amplitude values

---

### Dataset Source

**Paper**

Schäfer, J., Barrsiwal, B. R., Kokhkharova, M., Adil, H., & Liebehenschel, J. (2021). *Human Activity Recognition Using CSI Information with Nexmon*. Applied Sciences, 11(19), 8860.

DOI: https://doi.org/10.3390/app11198860

**Official Dataset**

DOI: https://doi.org/10.21227/xr6j-0255

---

### Dataset Availability

The dataset is **not included** in this repository due to its size (approximately **550 MB**).

After downloading the dataset from the official source, place all CSV files inside:

```text
data/
└── csv/
```

The project expects all CSV files to be located in this directory before running the preprocessing and training pipeline.

## 🎯 Objectives

- Preprocess Wi-Fi CSI amplitude data
- Extract statistical features
- Train and evaluate machine learning models
- Compare classical ML and deep learning approaches
- Analyze model performance using standard evaluation metrics

---

## 🤖 Implemented Models

### Machine Learning
- k-Nearest Neighbors (k-NN)
- Support Vector Machine (SVM)

### Deep Learning
- CNN
- LSTM
- GRU (trained separately via `experiments/final_Gru.py`)
- CNN + LSTM
- CNN + BiLSTM

---

## 📂 Project Structure

```text
wifi-csi-human-activity-recognition/

├── data/
├── evaluation/
├── Exploratory Data Analysis/
├── experiments/
├── features/
├── models/
├── preprocessing/
├── results/

├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/spideytwenty99/wifi-csi-human-activity-recognition.git
cd wifi-csi-human-activity-recognition
```

All reported results were produced with **Python 3.12**. The pinned package
versions require Python 3.11 or newer — on older interpreters the installation
fails with `No matching distribution found`.

Create a virtual environment and install the required packages

```bash
python3.12 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Download the dataset as described under [Dataset Availability](#dataset-availability)
and place the CSV files in `data/csv/`. The dataset is **not** part of this
repository, and the pipeline will not run without it.

---

## ▶️ Running the Project

All commands must be run from the repository root, because the data path is
resolved relative to the working directory.

Run the model comparison — k-NN, SVM, LSTM, CNN, CNN-LSTM and CNN-BiLSTM.
Metrics and figures are written to `results/`.

```bash
python main.py
```

Run the stacked GRU. It is not part of `main.py` and has to be started
separately.

```bash
python experiments/final_Gru.py
```

Optional — the hyperparameter searches behind the reported configurations:

```bash
python experiments/cnn1D_TuningV2.py   # CNN grid search
python experiments/main_gru.py         # GRU grid search
```

---

## 📊 Dataset

The project uses Wi-Fi CSI amplitude measurements collected for indoor human activity recognition.

Activities include:

- Empty
- Lying
- Sitting
- Standing
- Walking



---

## 📈 Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📄 License

This repository is intended for educational and research purposes.