# NeuroMedic
Overview

NeuroMedic is a deep learning model designed for automated brain tumor detection using MRI scans. It leverages a Convolutional Neural Network (CNN) to classify MRI images into tumor and non-tumor categories, achieving high accuracy through preprocessing and data augmentation.

Features

Custom CNN Architecture optimized for binary classification.

Preprocessing Techniques including resizing, normalization, and data augmentation.

Performance Metrics such as accuracy, confusion matrix, precision, recall, and ROC curve.

Efficient Training using the Adam optimizer and cross-entropy loss.

Installation

Clone the Repository

git clone https://github.com/yourusername/NeuroMedic.git
cd NeuroMedic

Install Dependencies

pip install -r requirements.txt

Dataset

This project uses the "Brain Tumor Multimodal Image CT and MRI" dataset from Kaggle, consisting of T1-weighted and T2-weighted MRI scans labeled as tumor or non-tumor.

Usage

Train the Model

python train.py

Evaluate the Model

python evaluate.py

Model Architecture

Input Layer: 224x224 resized MRI images.

Feature Extraction Layers:

Conv2D (3x3) → ReLU → Conv2D (3x3) → ReLU → MaxPooling (2x2)

Conv2D (3x3) → ReLU → Conv2D (3x3) → ReLU → MaxPooling (2x2)

Conv2D (3x3) → ReLU → Conv2D (3x3) → ReLU → MaxPooling (2x2)

Fully Connected Layers:

Flatten → Fully Connected (256 neurons) → ReLU → Dropout (0.5)

Fully Connected (2 neurons) → Softmax Activation

Training Parameters

Loss Function: Cross-Entropy Loss

Optimizer: Adam (learning rate = 0.001)

Batch Size: 32

Epochs: 100

Device: GPU (if available)

Evaluation Metrics

Accuracy: Measures overall correctness.

Confusion Matrix: Visualizes true/false positives and negatives.

Precision, Recall, F1-Score: Provides insight into model reliability.

ROC Curve & AUC Score: Assesses classification confidence.

Results

NeuroMedic achieves approximately 95% accuracy in distinguishing between healthy and tumorous MRI scans, demonstrating its effectiveness in automated medical diagnosis.

Future Improvements

Expand dataset for better generalization.

Implement transfer learning with pre-trained networks.

Explore Vision Transformers (ViTs) for enhanced performance.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

Kaggle for the dataset.

Open-source AI and deep learning communities.

Contact

For questions or collaborations, reach out at [your email] or visit [your GitHub profile].
