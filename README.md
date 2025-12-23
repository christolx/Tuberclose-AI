# 🩺 Tuberculosis Detection System using ResNet50 (Transfer Learning)

Sistem deteksi tuberkulosis (TB) berbasis **Deep Learning** yang menggunakan **Transfer Learning ResNet50** dengan pendekatan **fine-tuning** untuk membantu skrining awal TB dari **Chest X-ray (CXR)** secara cepat dan konsisten melalui aplikasi web.

---

## 📌 Latar Belakang

Tuberkulosis (TB) masih menjadi masalah kesehatan serius, khususnya di Indonesia.  
Interpretasi X-ray dada secara manual:

- Sangat subjektif  
- Bergantung pada pengalaman radiolog  
- Tidak merata di daerah terpencil  
- Rentan kesalahan pada kasus TB tahap awal  

Proyek ini membangun **Computer-Aided Diagnosis (CAD)** berbasis AI untuk membantu tenaga medis melakukan **skrining awal TB** secara objektif dan efisien.

---

## 🎯 Tujuan Proyek

- Mengembangkan model CNN berbasis **ResNet50 pretrained**
- Menerapkan **transfer learning + fine-tuning**
- Klasifikasi **Normal vs Tuberculosis**
- Integrasi ke **aplikasi web**
- Optimasi **recall TB** (minim false negative)

---

## 🧠 Teknologi & Tools

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

- Sumber: Kaggle – Chest X-ray Tuberculosis Dataset
- Kelas:
  - Normal
  - Tuberculosis
- Split data:
  - 80% Training
  - 20% Validation (Stratified)

---

## 🔄 Preprocessing & Augmentasi

- Resize → 256 px  
- Center Crop → 224 × 224  
- Normalisasi ImageNet  
- Random Horizontal Flip  
- Random Rotation (±10°)  
- Color Jitter (brightness & contrast)  
- WeightedRandomSampler untuk class imbalance  

Tetap menggunakan **RGB (3 channel)** untuk menjaga detail tekstur paru-paru.

---

## 🏗️ Arsitektur Model

- Base model: **ResNet50**
- Layer awal dibekukan
- Layer 3, Layer 4, dan Fully Connected di-fine-tune

### Training Setup
- Optimizer: Adam  
- Learning rate: `1e-4`  
- Max epoch: 30  
- Scheduler: ReduceLROnPlateau  
- Early Stopping  
- Kill-switch jika training accuracy > 99.5%

---

## 📊 Hasil Evaluasi (External Dataset)

### Model Kami (PyTorch)

- Accuracy: **93.38%**
- TB Recall: **95.63%**
- F1-Score (TB): **0.9600**
- ROC AUC: **0.9666**

### Model Pembanding (Keras)

- Accuracy: 91.69%
- TB Recall: 92.46%
- F1-Score (TB): 0.9486
- ROC AUC: 0.9566

---

## 📉 Confusion Matrix (External Data)

| Kategori | Model Kami | Model Lain |
|--------|-----------|-----------|
| True Normal (TN) | 424 | 452 |
| False Positive (FP) | 90 | 62 |
| False Negative (FN) | **109** | **188** |
| True TB (TP) | **2385** | 2306 |

Dalam konteks medis, **False Negative jauh lebih berbahaya**, dan model ini unggul dalam menekan FN.

---

## 🌐 Sistem Berbasis Web

Alur penggunaan:
1. Upload X-ray dada
2. Backend preprocessing & inferensi
3. Output:
   - Klasifikasi (Normal / TB)
   - Confidence score

Tidak perlu instalasi tambahan di sisi user.

---

## 🚀 Pengembangan Lanjutan

- Dataset multi-device X-ray
- Multi-class classification (TB vs pneumonia, dll)
- Integrasi Sistem Informasi Rumah Sakit (SIRS)
- Aplikasi mobile
- Uji klinis lapangan

---

## 👨‍💻 Tim Pengembang

- Christofle Tjhai  
- Nicholas Wilson Andrean  
- Jodie Obadja  

BINUS University

---

## ⚠️ Disclaimer

Sistem ini **bukan pengganti diagnosis dokter**, melainkan **alat bantu skrining awal**.
