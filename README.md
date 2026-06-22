BTC-MAP: Brain Tumor Classification Framework

This repository provides a simplified implementation of the BTC-MAP
framework for MRI-based brain tumor classification.

Pipeline
1. MRI preprocessing
2. Feature extraction (statistical + GLCM texture)
3. PCA dimensionality reduction
4. Classification using SVM

Dataset Structure

dataset/
    benign/
    malignant/

Example

dataset/
    benign/image1.png
    benign/image2.png
    malignant/image3.png
    malignant/image4.png

Run

python main.py

Reproducibility Note

This repository provides a simplified implementation of the BTC-MAP framework
to demonstrate the pipeline described in the research paper.

Public MRI datasets such as Figshare Brain MRI Dataset, Kaggle Brain MRI
Dataset, or BRATS may be used with this pipeline.

The code is released for research transparency and reproducibility.

Code Availability

The implementation of the BTC-MAP framework is publicly available
to support transparency and reproducibility of the research.

Repository:
https://github.com/yourusername/BTC-MAP

