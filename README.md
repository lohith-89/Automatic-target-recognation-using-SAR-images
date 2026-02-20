🚀 SAR Automatic Target Recognition (ATR) System
📌 Project Overview

This project implements a Deep Learning–based Automatic Target Recognition (ATR) system for classifying military vehicles from Synthetic Aperture Radar (SAR) images.

The system performs:

SAR image preprocessing

Edge enhancement using Sobel filters

Multi-scale CNN classification

Confidence-based prediction handling

Threat assessment generation

Web-based deployment using Flask

The system achieves high accuracy and provides real-time prediction via a web interface.

🧠 Technologies Used

Python

TensorFlow / Keras

Flask

NumPy

Scikit-learn

Matplotlib

Bootstrap (Frontend)

SQLite (Database)

📂 Project Structure
SAR/
│
├── app1.py                        # Main Flask application
├── downstream_model_weights.h5    # Trained ATR model weights
├── self_supervised_model_weights.weights.h5
├── model.h5
├── model2.h5
│
├── templates/                     # HTML pages
├── static/                        # CSS, JS, images
│
├── data/                          # Training dataset
├── Unlabeled/                     # Unlabeled SAR images
├── alerts/                        # Alert logs
├── reports/                       # Generated reports
├── flagged/                       # Flagged images
│
├── sar_system.db                  # Database file
├── alarms.json
├── analysis_results.json
├── users.json
│
├── requirements.txt
├── runtime.txt
├── render.yaml
├── Procfile
└── README.md
🏗 System Workflow

1️⃣ User uploads SAR image
2️⃣ Image is validated
3️⃣ Preprocessing (Resize + Normalize)
4️⃣ Sobel Edge Extraction
5️⃣ Multi-Scale CNN Classification
6️⃣ Confidence Score Calculation
7️⃣ Threat Level Assignment
8️⃣ Alert & Result Storage

🧩 Model Details

Input size: 224 × 224 × 3

Sobel edge-based feature enhancement

Multi-scale convolutions (3×3, 5×5, 7×7)

Optimizer: Adam

Loss: Categorical Crossentropy

Output: Softmax probabilities

📊 Evaluation Metrics

The system is evaluated using:

Accuracy curves

ROC curves (AUC)

Confusion matrix

Precision, Recall, F1-score

Model achieves ~99% accuracy on test dataset.

🚨 Threat Assessment

Detected targets are categorized into:

Low

Medium

High

Critical

Low-confidence predictions are marked as:

Unknown
🌐 Running the Project Locally
1️⃣ Clone Repository
git clone https://github.com/your-username/sar-atr.git
cd SAR
2️⃣ Install Requirements
pip install -r requirements.txt
3️⃣ Run Flask App
python app1.py

Open in browser:

http://127.0.0.1:5000/
☁️ Deployment

This project supports deployment on:

Render (render.yaml included)

Any WSGI-supported server

🔐 Features

✔ Confidence Thresholding
✔ SAR Image Validation
✔ Alert Generation
✔ Report Storage
✔ Database Integration
✔ Web Interface

👨‍💻 Authors

Amruth K S
Lohith R

B.E. CSE (Data Science)
SJB Institute of Technology
VTU