# Sign Language Recognition System: ML & DL Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Mission](#project-mission)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)

---

## 🎯 Overview

This project demonstrates a comprehensive machine learning and deep learning pipeline for sign language gesture recognition, aimed at developing assistive technologies for people with disabilities. The analysis focuses on comparing traditional ML algorithms and DL architectures to enable real-time hand gesture recognition for communication aids.

**Multiple experiments** are conducted, comparing:
- **3 Traditional ML Models**: Logistic Regression, Random Forest, Support Vector Machine (SVM)
- **2 Deep Learning Models**: Simple Convolutional Neural Network (CNN), Deeper CNN with hyperparameter tuning

---

## 🌍 Project Mission

> **To empower people with disabilities through accessible AI-driven gesture recognition systems, making sign language communication more inclusive and affordable, especially in resource-limited settings like developing regions.**

This research operationalizes that mission by:
1. **Evaluating model performance** for accuracy and efficiency
2. **Optimizing for low-compute devices** to ensure deployability
3. **Addressing class imbalances** and overfitting challenges
4. **Providing visualizations** like confusion matrices and ROC curves for interpretability
5. **Emphasizing social impact** through hybrid ML-DL approaches for accessibility

---

## 📊 Dataset

**Source:** [Sign Language MNIST from Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

| Metric                  | Value           |
| ----------------------- | --------------- |
| **Training Samples**    | 27,455          |
| **Test Samples**        | 7,172           |
| **Image Size**          | 28x28 grayscale |
| **Classes**             | 24 (A-Y in ASL) |
| **Missing Values**      | 0               |
| **Format**              | CSV (pixels)    |

### Features
- **Pixel Values**: 784 features (flattened 28x28 images)
- **Labels**: 0-24 representing letters A-Y (excluding J and Z due to motion)

### Preprocessing
- Resizing and normalization for DL models
- HOG feature extraction for ML models
- Train-validation-test split

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Step 1: Clone Repository
```bash
git clone https://github.com/Q-Reine/Sign_Language_Recognition.git
cd Sign_Language_Recognition
```

### Step 2: Verify Installation
```bash
python -c "import tensorflow, sklearn, pandas; print('✓ All packages installed successfully')"
```

---

## 🚀 Quick Start

### Run the Complete Analysis

```bash

# Start Jupyter notebook
jupyter notebook sign_language_recognition.ipynb

# Or use Google Colab (no installation needed)
# Open: https://colab.research.google.com/github/Q-Reine/Sign_Language_Recognition.git/sign_language_recognition.ipynb
```

### Output
The notebook generates:
- ✓ ML and DL model results
- ✓ Accuracy, precision, recall, F1-scores
- ✓ 10+ visualizations (learning curves, confusion matrices, ROC curves)
- ✓ Feature importance (where applicable)
- ✓ Hyperparameter tuning results
- ✓ Comprehensive comparison tables

---

## 📁 Project Structure

```
assistive-gesture-recognition/
├── README.md                                            # This file
├── sign_language_recognition.ipynb                      # Main analysis notebook
```

---

## 🔬 Methodology

### Approach

**Stage 1: Data Preparation**
- Load dataset from Kaggle CSV files
- Handle any imbalances or preprocessing needs
- Extract HOG features for ML
- Reshape images for DL (28x28x1)
- Normalize pixel values
- Train-validation-test split with random_state=42

**Stage 2: Traditional ML Models (3 experiments)**
- Logistic Regression with GridSearchCV
- Random Forest with GridSearchCV
- Support Vector Machine (SVM) with GridSearchCV
- Evaluation on flattened/HOG features

**Stage 3: Deep Learning Models (2 experiments)**
- Simple CNN: Conv2D, MaxPooling, Dense layers
- Deeper CNN: Tuned with Keras Tuner (conv units, dense units, dropout, learning rate)
- EarlyStopping callback
- Categorical cross-entropy loss

**Stage 4: Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-score
- Visualizations: Learning curves, confusion matrices, multi-class ROC curves
- Cross-model comparison for interpretability vs. performance
- Insights on overfitting and resource efficiency

---

## 🔄 Reproducibility

**Full Reproducibility Guaranteed:**
- ✅ Random seed set to 42 (NumPy & TensorFlow)
- ✅ Dataset sourced from public Kaggle URL
- ✅ All preprocessing documented
- ✅ Hyperparameters explicitly listed
- ✅ No hardcoded paths or dependencies
- ✅ Notebook runs top-to-bottom without errors
- ✅ Runtime: ~20-30 minutes on standard CPU

To reproduce:

```bash
jupyter notebook sign_language_recognition.ipynb
# Run all cells (Ctrl+A, then Ctrl+Enter)
# Exact results will be generated
```
---

## 🙏 Acknowledgments

- **Dataset:** [Sign Language MNIST Contributors on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Libraries:** TensorFlow, Scikit-learn, Pandas, Matplotlib, Seaborn, Keras Tuner, Scikit-image
- **Inspiration:** Assistive technologies for disability inclusion
- **Project Mission:** Making AI accessible for sign language recognition

---

**Last Updated:** 19th October 2025  
**Reproducibility:** ✅ Fully Reproducible (seed=42)

---
