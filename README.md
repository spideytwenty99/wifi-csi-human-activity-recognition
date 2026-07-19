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
- GRU
- CNN + LSTM
- CNN + BiLSTM

---

## 📂 Project Structure

```text
wifi_activity_recognition/

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
git clone https://github.com/YOUR_USERNAME/wifi-activity-recognition.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the main script

```bash
python main.py
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