# 🩺 Breast Cancer Classification using Artificial Neural Networks (ANN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

## 📌 Overview
An end-to-end Deep Learning diagnostic model utilizing **Artificial Neural Networks (ANN)** to predict whether a breast tumor is **Malignant** or **Benign** based on cell nucleus characteristics from the Wisconsin Diagnostic Breast Cancer dataset.

---

## 🧠 Model Architecture & Pipeline

```text
Input Layer (30 features)
       │
       ▼
Dense Layer (30 units, ReLU activation)
       │
       ▼
Dense Layer (16 units, ReLU activation)
       │
       ▼
Dropout Layer (0.2 regularization)
       │
       ▼
Dense Layer (8 units, ReLU activation)
       │
       ▼
Output Layer (1 unit, Sigmoid activation) ──► Probability [0.0 - 1.0] (Malignant / Benign)
```

- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy, Precision, Recall, ROC-AUC

---

## 📊 Dataset Features
The model trains on 30 diagnostic numerical attributes computed from digitized fine needle aspirate (FNA) images, including:
- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity, Concave points
- Symmetry, Fractal dimension

---

## 🚀 Quickstart & Usage

### 1. Prerequisites
```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Run the Notebook
```bash
jupyter notebook Cancer_Classification_using_ANN.ipynb
```

---

## 👨‍💻 Author
**Muhammad Okasha**  
[GitHub Profile](https://github.com/muhammadokashapak)
