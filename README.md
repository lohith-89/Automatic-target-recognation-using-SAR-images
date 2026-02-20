# 🚀 SAR Automatic Target Recognition (ATR) System

## 📌 Project Overview

This project implements a **Deep Learning–based Automatic Target Recognition (ATR) system** for classifying military vehicles from **Synthetic Aperture Radar (SAR) images**.

Synthetic Aperture Radar (SAR) works in all weather and lighting conditions, making it highly useful in defense and surveillance applications. This system automates the detection and classification process using deep learning techniques to improve accuracy, reliability, and speed.

---

## 🎯 Objectives

- Develop a deep learning–based ATR system for accurate SAR image classification  
- Reduce dependency on large labeled datasets  
- Apply confidence-based prediction handling  
- Perform automated threat level assessment  
- Deploy a real-time web-based system  

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
2. SAR image validation  
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

## 📂 Project Structure

```
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
```

---

## ⚙️ Installation & Setup

### Clone Repository

```bash
git clone https://github.com/lohith-89/Automatic-target-recognation-using-SAR-images.git
cd Automatic-target-recognation-using-SAR-images
```

### Create Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate environment:

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements file is missing:

```bash
pip install flask tensorflow numpy scikit-learn matplotlib pillow gdown flask-cors
```

### Run the Application

```bash
python app1.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🚨 Threat Assessment

Based on the recognized vehicle type, the system assigns threat levels:

- Low  
- Medium  
- High  
- Critical  

Low-confidence predictions are marked as **Unknown** to minimize false alarms.

---



---

## 👨‍💻 Authors

### Lohith R  
B.E. Computer Science and Engineering (Data Science)  
SJB Institute of Technology  
Visvesvaraya Technological University (VTU)  
Academic Year: 2025–26  

### Amruth K S  
B.E. Computer Science and Engineering (Data Science)  
SJB Institute of Technology  
Visvesvaraya Technological University (VTU)  
Academic Year: 2025–26  

---

## 📜 License

This project is developed for academic and research purposes under VTU guidelines.