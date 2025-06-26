# Machine Learning Projects Portfolio

This repository contains multiple machine learning projects implemented using Python and TensorFlow/Keras. Each project focuses on a different classification task with relevant datasets and models. These projects demonstrate skills in data preprocessing, model building, hyperparameter tuning, and evaluation.

---

## Projects Overview

### 1. MNIST Digit Classification

- **Description:** Neural network to classify handwritten digits (0-9) using the MNIST dataset.
- **Dataset:** 70,000 grayscale 28x28 images of digits.
- **Model:** Fully connected neural network with dropout and ReLU activation.
- **Results:** Achieved ~98% test accuracy. Explored effects of tuning epochs, dropout, nodes, and batch size.
- **Notebook:** [`MNIST_Classification.ipynb`](./MNIST_Classification.ipynb)

---

### 2. CIFAR-10 Image Classification

- **Description:** CNN model classifying 32x32 color images into 10 categories.
- **Dataset:** CIFAR-10 with 60,000 images across 10 classes.
- **Model:** Convolutional Neural Network with multiple Conv2D, MaxPooling, and dropout layers.
- **Results:** Achieved 80% test accuracy with model2. Experimented with architecture and regularization.
- **Notebook:** [`CIFAR_Classification.ipynb`](./CIFAR_Classification.ipynb)

---

### 3. Reuters Newswire Classification

- **Description:** Multi-class text classification of Reuters news articles into 46 topics.
- **Dataset:** 11,228 news articles with topic labels.
- **Model:** Multi-layer perceptron with two dense layers and dropout.
- **Results:** Achieved ~70% test accuracy. Addressed challenges with class imbalance.
- **Notebook:** [`Reuters_Classification.ipynb`](./Reuters_Classification.ipynb)

---

### 4. Bamboo Detection from Satellite Images

- **Description:** CNN model to classify satellite images as containing bamboo forests or not.
- **Dataset:** Custom dataset with labeled satellite images (bamboo vs. non-bamboo). Images are divided into overlapping 56×56 patches for training.
- **Model:** Convolutional Neural Network with 3 Conv2D layers, MaxPooling, Dropout, and Dense layers.
- **Results:** Achieved strong binary classification performance. Addressed class imbalance using weighted loss. Visualized predictions and explored effects of patching.
- **Notebook:** [`Bamboo_Detection.ipynb`](./Bamboo_Detection.ipynb)

---


## Getting Started

### Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

You can install dependencies with:

```bash
pip install tensorflow numpy matplotlib jupyter
