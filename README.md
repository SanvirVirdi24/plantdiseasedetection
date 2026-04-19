# 🌿 Plant Disease Classification Project

A lightweight computer vision workflow built using Python to identify and classify plant leaf diseases efficiently.

---

## 🚀 Overview

This project implements a simple yet effective deep learning pipeline for plant disease classification. It uses a Convolutional Neural Network (CNN) along with image preprocessing and OpenCV-based segmentation to deliver accurate predictions.

---

## ✨ Key Features

- 📂 **Automatic Dataset Distribution**
  - Splits dataset into training (80%) and testing (20%)

- 🖼️ **Image Preprocessing**
  - Normalizes pixel values for improved performance

- 🧠 **Simplified CNN Architecture**
  - Conv2D → ReLU → MaxPooling → Dense → Dropout
 

- 🌱 **Leaf Color Segmentation (OpenCV)**
- Uses HSV masking to isolate leaf regions

- 📊 **Performance Evaluation**
- Accuracy & Loss graphs
- Confusion Matrix
- Epoch tracking

- 🔍 **Prediction System**
- Random test image prediction
- Custom image input support
- Confidence visualization

---

## 📁 Project Structure
```
.
├── dataset/                    # Generated folder holding separated 'train' and 'test' images
├── split_data.py               # Utility script to construct the dataset layout
├── train.py                    # Core pipeline to build, train, evaluate, and save CNN model
├── predict.py                  # Utility script to process segmentations and display predictions
├── models/                     # Output repository where the network state (h5) and configurations save
├── prediction_output.png       # Generated visualization grid
├── training_graphs.png         # Generated metrics for model evaluation
└── confusion_matrix.png        # Generated heatmap
```

---

## ⚙️ Setup & Usage

### 1️⃣ Dataset Setup

Dataset is already prepared using:

```bash
python3 split_data.py
```
### 2️⃣ Train the Model

```bash
python3 train.py
```
### Run Inference and Mask Leaf Outcomes
To test out the deployed model logic, execute predict.py. Have a specific leaf image to inspect? Pass it logically as an argument!

# Run against a random testing image
```bash
python3 predict.py 
```
# Or test a specific customized photo!
```bash
python3 predict.py dataset/test/Potato___Early_blight/example_image.jpg
```
Creates the CLI output text as well drawing the visualization output to prediction_output.png.

Developed using standard ML setups: tensorflow, opencv-python, scikit-learn, matplotlib, and seaborn.
