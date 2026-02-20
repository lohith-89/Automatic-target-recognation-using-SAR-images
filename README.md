# 🚀 SAR Automatic Target Recognition (ATR) System

## 📌 Project Overview

This project implements a **Deep Learning–based Automatic Target Recognition (ATR) system** for classifying military vehicles from **Synthetic Aperture Radar (SAR) images**.

Synthetic Aperture Radar (SAR) works in all weather and lighting conditions, making it highly useful in defense and surveillance applications.  
This system automates the detection and classification process using deep learning techniques to improve accuracy, reliability, and speed.

The system integrates edge enhancement, multi-scale convolution, confidence-based prediction handling, and automated threat assessment within a real-time web application.

---

## 🎯 Objectives

- Develop a deep learning–based ATR system for accurate SAR image classification  
- Reduce dependency on large labeled datasets using effective feature learning  
- Apply confidence-based prediction handling to reduce false classifications  
- Perform automated threat level assessment based on detected targets  
- Deploy a real-time web-based system for practical usage  

---

## 🧠 Technologies Used

- Python  
- TensorFlow / Keras  
- Flask  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Bootstrap  
- SQLite  

---

## 🏗 System Workflow

1. User uploads SAR image  
2. SAR image validation is performed  
3. Image preprocessing (Resize + Normalize)  
4. Sobel edge feature extraction  
5. Multi-scale CNN classification  
6. Confidence score calculation  
7. Threat level assignment (Low / Medium / High / Critical)  
8. Alert generation and result storage  

---

## 🧩 Model Architecture

- Input Size: 224 × 224 × 3  
- Edge Enhancement: Sobel Filter  
- Multi-scale Convolutions: 3×3, 5×5, 7×7  
- Activation Function: ReLU  
- Output Layer: Softmax  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  

---

## 📊 Performance Evaluation

The model performance is evaluated using:

- Training and validation accuracy curves  
- ROC Curve (AUC Score)  
- Confusion Matrix  
- Precision, Recall, F1-score  

The system achieves approximately **99% classification accuracy** on the test dataset.

---

## 🚨 Threat Assessment

Based on the recognized vehicle type, the system assigns threat levels:

- Low  
- Medium  
- High  
- Critical  

Low-confidence predictions are marked as **Unknown** to minimize false alarms and improve system reliability.

---

## 🌐 Web Application Features

- Secure SAR image upload  
- Real-time prediction  
- Confidence score display  
- Threat level classification  
- Alert generation  
- Result storage and reporting  
- Database integration  

---

## 📂 Project Structure

SAR/
│
├── app1.py  
├── downstream_model_weights.h5  
├── model.h5  
├── model2.h5  
├── self_supervised_model_weights.weights.h5  
│  
├── templates/  
├── static/  
│  
├── data/  
├── Unlabeled/  
├── alerts/  
├── reports/  
├── flagged/  
│  
├── sar_system.db  
├── alarms.json  
├── analysis_results.json  
├── users.json  
│  
├── requirements.txt  
├── runtime.txt  
├── render.yaml  
├── Procfile  
└── README.md  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/lohith-89/Automatic-target-recognation-using-SAR-images.git
cd SAR


## 👨‍💻 Authors

### 🔹 Lohith R  
B.E. Computer Science and Engineering (Data Science)  
SJB Institute of Technology  
Visvesvaraya Technological University (VTU)  
Academic Year: 2025–26  

### 🔹 Amruth K S  
B.E. Computer Science and Engineering (Data Science)  
SJB Institute of Technology  
Visvesvaraya Technological University (VTU)  
Academic Year: 2025–26  