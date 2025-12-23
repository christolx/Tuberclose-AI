# 🩺 Tuberculosis Detection System using ResNet50 (Transfer Learning)

An AI-based tuberculosis (TB) detection system utilizing **Deep Learning** with **ResNet50 Transfer Learning** and **fine-tuning** to support early TB screening from **Chest X-ray (CXR)** images through a web-based application.

---

## 📌 Background

Tuberculosis (TB) remains a major global health problem, especially in developing countries like Indonesia. Manual interpretation of chest X-rays:

- Is highly subjective  
- Depends heavily on radiologist experience  
- Suffers from limited availability of specialists  
- Is prone to missed early-stage TB cases  

This project aims to build an **AI-powered Computer-Aided Diagnosis (CAD)** system to assist healthcare workers in performing **fast, consistent, and objective TB screening**.

---

## 🎯 Project Objectives

- Develop a CNN model based on **pretrained ResNet50**
- Apply **transfer learning and fine-tuning**
- Perform binary classification (**Normal vs Tuberculosis**)
- Integrate the model into a **web-based system**
- Optimize **TB recall** to minimize false negatives

---

## 🧠 Technologies & Tools

### Machine Learning
- PyTorch
- ResNet50 (ImageNet pretrained)
- Transfer Learning
- Fine-Tuning
- Test Time Augmentation (TTA)
- Threshold Optimization

### Web Development
- Frontend: React JS, Tailwind CSS
- Backend: FastAPI (Python)
- Model format: `.pth`

---

## 🗂️ Dataset

- Source: Kaggle – Chest X-ray Tuberculosis Dataset
- Classes:
  - Normal
  - Tuberculosis
- Data split:
  - 80% Training
  - 20% Validation (Stratified)

---

## 🔄 Preprocessing & Augmentation

- Resize to 256 px  
- Center Crop to 224 × 224  
- ImageNet normalization  
- Random Horizontal Flip  
- Random Rotation (±10°)  
- Color Jitter (brightness & contrast)  
- WeightedRandomSampler for class imbalance  

All images are kept in **RGB (3 channels)** to preserve subtle lung texture details critical for TB detection.

---

## 🏗️ Model Architecture

- Base model: **ResNet50**
- Early layers frozen
- Layer 3, Layer 4, and Fully Connected layers fine-tuned

### Training Configuration
- Optimizer: Adam  
- Learning rate: `1e-4`  
- Maximum epochs: 30  
- Scheduler: ReduceLROnPlateau  
- Early Stopping  
- Training kill-switch if accuracy exceeds 99.5%

---

## 📊 Evaluation Results (External Dataset)

### Our Model (PyTorch)

- **Accuracy**: 93.38%
- **TB Recall**: 95.63%
- **F1-Score (TB)**: 0.9600
- **ROC AUC**: 0.9666

### Baseline Model (Keras)

- Accuracy: 91.69%
- TB Recall: 92.46%
- F1-Score (TB): 0.9486
- ROC AUC: 0.9566

---

## 📉 Confusion Matrix (External Data)

| Category | Our Model | Baseline Model |
|--------|-----------|----------------|
| True Normal (TN) | 424 | 452 |
| False Positive (FP) | 90 | 62 |
| False Negative (FN) | **109** | **188** |
| True TB (TP) | **2385** | 2306 |

In medical diagnostics, **false negatives are significantly more dangerous** than false positives. This model demonstrates superior performance in minimizing missed TB cases.

---

## 🌐 Web-Based System

Workflow:
1. User uploads a chest X-ray image
2. Backend performs preprocessing and inference
3. Output:
   - Classification result (Normal / Tuberculosis)
   - Prediction confidence score

No additional software installation is required on the user side.

---

## 🚀 Future Improvements

- More diverse datasets from multiple X-ray devices
- Multi-class classification (TB vs pneumonia vs others)
- Integration with Hospital Information Systems (HIS)
- Mobile application development
- Clinical field testing in healthcare facilities

---

## 👨‍💻 Development Team

- Christofle Tjhai  
- Nicholas Wilson Andrean  
- Jodie Obadja  

BINUS University

---

## ⚠️ Disclaimer

This system is **not a replacement for professional medical diagnosis**.  
It is intended as a **decision-support tool for early TB screening** only.
