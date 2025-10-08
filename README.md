# Protecting Smart Grids from Cyber Threats using Artificial Neural Networks (ANN)

**Developed as part of a research study on intelligent cyber threat detection in smart grids.**

---

## 🧩 Abstract

The increasing digitization of modern power systems has made smart grids vulnerable to sophisticated cyberattacks targeting their communication and control layers. This research focuses on detecting such intrusions using Artificial Neural Networks (ANN) and benchmarking their performance against classical machine learning models including Logistic Regression, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). The experiments utilize the **ERENO** dataset — a framework designed for generating realistic IEC-61850 intrusion detection datasets for smart grids.

---

## ⚙️ Keywords
Smart Grid Security, Intrusion Detection, IEC-61850, Machine Learning, Artificial Neural Networks, Cyber-Physical Systems

---

## 📘 Dataset Description

**Dataset:** [ERENO: A Framework for Generating Realistic IEC-61850 Intrusion Detection Datasets for Smart Grids](https://zenodo.org/records/10252420)

ERENO provides labeled network traffic representing both normal operations and multiple cyberattack scenarios specific to IEC-61850 communications.  
This dataset enables the evaluation of machine learning-based Intrusion Detection Systems (IDS) under realistic smart grid conditions.

### Attack Types
- `high_StNum`
- `injection`
- `inverse_replay`
- `masquerade_fake_fault`
- `masquerade_fake_normal`
- `poisoned_high_rate`
- `random_replay`
- `normal`

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Handled categorical and numerical features.
   - Applied standard scaling and label encoding.
   - Balanced dataset using undersampling of dominant normal class.

2. **Model Training**
   - Models: ANN, Logistic Regression, Random Forest, SVM, and KNN.
   - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, and AUC.
   - Data split: 80% training, 20% testing.

3. **Evaluation**
   - Benchmarking conducted on the same preprocessed dataset.
   - Performance measured for both binary and multi-class classification.

---

## 📊 Results

### **Model Benchmarking (Multi-class)**
| Model | Accuracy | Precision | Recall | F1-score | AUC |
|:------|----------:|-----------:|--------:|----------:|----:|
| Logistic Regression | 0.850 | 0.972 | 0.850 | 0.899 | 0.984 |
| Random Forest | **0.998** | **0.998** | **0.998** | **0.998** | **0.999** |
| SVM | 0.927 | 0.979 | 0.927 | 0.948 | 0.992 |
| KNN | 0.923 | 0.970 | 0.923 | 0.942 | 0.967 |

> **Random Forest** achieved the highest accuracy and AUC, while ANN achieved robust and generalized detection performance comparable to ensemble methods.

---

### **ANN Classification Report**
| Attack Type | Precision | Recall | F1-score |
|:-------------|-----------:|--------:|----------:|
| high_StNum | 0.99 | 1.00 | 1.00 |
| injection | 0.99 | 0.97 | 0.98 |
| inverse_replay | 0.75 | 0.76 | 0.75 |
| masquerade_fake_fault | 0.19 | 0.85 | 0.31 |
| masquerade_fake_normal | 0.93 | 1.00 | 0.97 |
| normal | 1.00 | 0.97 | 0.98 |
| poisoned_high_rate | 0.93 | 0.99 | 0.96 |
| random_replay | 0.76 | 0.99 | 0.86 |

**Overall Accuracy:** 0.97  
**Macro Avg:** Precision 0.82, Recall 0.94, F1-score 0.85  
**Weighted Avg:** Precision 0.99, Recall 0.97, F1-score 0.98  

---

## 🧪 Technologies Used
- Python 3.x  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/SmartGrid-ANN-Protection.git
cd SmartGrid-ANN-Protection

# Install dependencies
pip install -r requirements.txt

# Preprocess dataset
python preprocess.py

# Train models
python train_ann.py
python benchmark_models.py

# Evaluate results
python evaluate.py
