# Comparative Deep Learning for PV Cell Defect Severity Classification

This repository contains the codebase and deliverables for the **ICT-4442 Deep Learning Mini Project**. The project focuses on automating the inspection of Photovoltaic (PV) cells by classifying structural defect severity from Electroluminescence (EL) imagery.

## 👥 Team Members
* **Atharva Agrawal** (230911348) - Data Pipeline, HOG+SVM Baseline
* **Ishita Gupta** (230911542) - Deep Learning Architectures (ResNet-18, ViT), Data Augmentation, Training Loops

## 📊 Dataset
We utilized the publicly available **Electroluminescence Photovoltaic (ELPV)** dataset.
* **Size**: 2,624 grayscale, 8-bit EL images ($300\times300$ pixels).
* **Classes**: 4 discrete integer classes mapped from continuous defect probabilities (0.0: None, 0.33: Low, 0.67: Medium, 1.0: High).
* **Handling Imbalance**: We enforced a strict 70/15/15 (Train/Val/Test) stratified split across all models to preserve the class distribution.

## 🧠 Models Implemented
This project compares three distinct machine learning paradigms to evaluate representational capacity vs. data starvation on small industrial datasets:

1. **HOG + SVM (Classical Baseline)**: Handcrafted feature extraction via Histograms of Oriented Gradients passed to a Support Vector Machine with an RBF kernel.
2. **ResNet-18 (CNN)**: ImageNet-pretrained convolutional network. Adapted for 1-channel grayscale inputs and trained with heavy data augmentation (rotations, flips) to prevent catastrophic overfitting.
3. **Vision Transformer (ViT-B/16)**: State-of-the-art attention mechanism. Images were interpolated to $224\times224$ and duplicated across 3 channels. Stabilized using Cosine Annealing, gradient clipping, and aggressive augmentation.

## 🏆 Final Results (Test Set)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| **HOG + SVM** | 68.0% | 53.0% | 57.0% | 54.0% |
| **ResNet-18 (Augmented)** | 61.0% | 50.0% | 57.0% | 51.0% |
| **Vision Transformer** | 64.0% | 56.0% | 61.0% | 54.0% |

**Key Insight:** While deep learning models inherently suffer from data starvation on datasets of this size, rigorous data augmentation and regularization allow attention-based mechanisms (ViT) to outperform standard convolutional hierarchies (ResNet) in capturing complex defect topologies.

## 🚀 How to Run

### 1. Local Execution (HOG+SVM)
Ensure you have the required dependencies (`scikit-learn`, `scikit-image`, `numpy`). Run the baseline script directly:
```bash
python baseline_hog_svm.py
```

### 2. Google Colab Execution (ResNet-18 & ViT)
Due to the computational intensity of the deep learning architectures, we recommend running the included Colab notebook:
1. Upload the dataset/repository to Google Drive.
2. Open `final_model_training.ipynb` in Google Colab.
3. Select a **T4 GPU** runtime.
4. Run all cells to execute the augmented training pipelines and generate evaluation metrics.

## 📚 References
* Buerhop-Lutz, C. et al. (2018). *A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery*. EU PVSEC.
* Deitsch, S. et al. (2019). *Automatic classification of defective photovoltaic module cells in electroluminescence images*. Solar Energy.