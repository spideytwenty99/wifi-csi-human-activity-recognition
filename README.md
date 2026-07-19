# WiFi CSI Human Activity Recognition

Project on **human activity recognition** using **Wi-Fi Channel State Information (CSI)** and machine learning/deep learning models.

---

## 📖 Project Overview

This project investigates the use of Wi-Fi CSI signals for recognizing indoor human activities without requiring wearable sensors or cameras.

Several classical machine learning and deep learning models are implemented and evaluated using the same dataset and preprocessing pipeline.

The project was developed as part of a Bachelor's thesis.

---

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