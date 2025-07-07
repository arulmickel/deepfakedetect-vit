# DeepfakeDetect: Vision Transformer Based Deepfake Image Classifier

## Overview
DeepfakeDetect is a neural network-based project focused on classifying images as real or deepfake using Vision Transformers (ViT). The project addresses the critical challenge of identifying manipulated media content, which has become increasingly sophisticated and widespread due to advancements in generative AI.

## Course Information
This project was developed for the **Neural Networks | CSC 578** course as part of the MS in Computer Science - Artificial Intelligence track at DePaul University.

## Abstract
With the rise of deepfake technology, verifying the authenticity of digital media has become a pressing issue. This project implements a Vision Transformer-based classification model trained on a balanced dataset of over 190,000 real and fake images. Using data augmentation, oversampling, and advanced training strategies, the system effectively learns to distinguish subtle visual differences between real and deepfake content.

## Key Features
- Fine-tuned pre-trained **ViT (Vision Transformer)** model
- Dataset sourced from **Celeb-DF (v2)** and **Kaggle**
- **Data augmentation** and **oversampling** to combat class imbalance
- Achieved **99.35% accuracy**, **99.30% F1 Score**, and **99.40% ROC AUC**
- Uses **AdamW optimizer** with **early stopping** and **cross-entropy loss**

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- imbalanced-learn
- Kaggle API
- Jupyter Notebook

## Project Structure
```
deepfakedetect-vit/
├── README.md
├── requirements.txt
├── Final Project Report.pdf
├── notebooks/
│   └── DeepfakeDetection_ViT.ipynb
├── scripts/
│   └── train_model.py (optional)
├── models/
│   └── best_model.pth (optional)
├── images/
│   └── sample_predictions.png
└── data/ (placeholder or add .gitignore)
```

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/deepfakedetect-vit.git
cd deepfakedetect-vit
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Open the notebook in Jupyter or run the training script (if provided).

## Results
The Vision Transformer model demonstrated superior performance, especially in distinguishing subtle visual inconsistencies typical of deepfake images. The final model achieved:
- **Accuracy:** 99.35%
- **F1 Score:** 99.30%
- **ROC AUC:** 99.40%

## 📄 Dataset
- Used a cleaned version of the [Kaggle Deepfake Detection Dataset](https://www.kaggle.com/competitions/deepfake-classification) (2 classes: `REAL`, `FAKE`)
- Images resized to 128x128 for efficient processing.

## Final Report
Please refer to the `Final Project Report.pdf` for complete details on methodology, results, and visual examples.

---
**Subject:** Neural Networks  
**University:** DePaul University, Chicago