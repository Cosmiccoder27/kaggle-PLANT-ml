# kaggle-PLANT-ml

## WIDS-2025-END-TO-END-ML
🌱 PlantVillage Disease Classification – End-to-End Pipeline

---

## 1. Project Overview
This project is an end-to-end machine learning pipeline for plant disease classification using the PlantVillage dataset. The work covers all stages of a real-world ML workflow:

- Dataset exploration, preprocessing, and integrity checks  
- Statistical analysis and handling of class imbalance  
- Classical machine learning baseline using handcrafted features  
- Deep learning using Convolutional Neural Networks (CNNs)  
- Transfer learning with pre-trained models for improved accuracy  
- Model evaluation and comparison across methods  
- Deployment-ready design, including considerations for federated learning  

The objective is to build a complete pipeline that starts from raw data and ends with a deployable ML model capable of accurate plant disease classification.

---

## 2. Dataset Description
The PlantVillage dataset contains images of healthy and diseased plant leaves across multiple crop species.  
The dataset is organized into three image representations:

- color  
- grayscale  
- segmented  

For this project, only color images are used.

---

## 3. Dataset Traversal and Metadata Creation
The dataset directory structure is programmatically traversed to collect metadata for each image.  
For every image, the following information is extracted:

- Image type (color / grayscale / segmented)  
- Disease class (derived from the folder name)  
- Image filename  

This information is stored in a pandas DataFrame.

---

## 4. Image Path Construction
A full absolute image path is constructed for every image by combining:

- Dataset root directory  
- Image type directory  
- Class directory  
- Image filename  

---

## 5. Class Distribution Analysis
The number of images in each disease class is computed.  
Observations:

- The dataset is highly imbalanced  
- Some disease classes contain more than 15,000 images  
- Other classes contain fewer than 500 images  
- Tomato and citrus diseases dominate

---

## 6. Visualization of Class Distribution
Class distributions are visualized using histograms and pie charts.

---

## 7. Image Size Analysis
The height and width of each image are extracted.  
Observation:

- All images have a resolution of 256 × 256 pixels

---

## 8. Corrupted Image Detection
Each image is checked for corruption.  
Result:

- 0 corrupted images found

---

## 9. Class Imbalance Considerations
To handle class imbalance:

- Stratified train–test splitting is used  
- Class distribution is preserved during evaluation

---

## 10. Project Workflow (Baseline Machine Learning Pipeline)
- Select color images only  
- Collect class labels and image paths  
- Convert class names to numerical labels using label encoding  
- Validate images to ensure no corruption  
- Extract HOG features  
- Split dataset: 80% training, 20% testing  
- Apply standardization  
- Apply PCA (retain 95% variance)  
- Train shallow ML model  
- Evaluate using accuracy and confusion matrix

---

## 11. Summary of Work Done
- Dataset traversal and metadata creation  
- Image path construction  
- Class distribution analysis  
- Visualization of class imbalance  
- Image size consistency verification  
- Corrupted image detection  
- Feature extraction using HOG  
- Dimensionality reduction using PCA  
- Baseline ML model training

---

## 12. Week 3: Deep Learning — CNN and Transfer Learning
- Train a simple CNN from scratch:  
  - Input: Rescaled images (128 × 128 × 3)  
  - Conv2D + ReLU + MaxPooling layers  
  - GlobalAveragePooling2D  
  - Dense layers → Softmax  
  - Optimizer: Adam  
  - Loss: Sparse Categorical Crossentropy  
  - Epochs: 4–5 (adjustable)  
- Evaluate accuracy and loss on training and validation datasets

- Apply Transfer Learning using pre-trained models (e.g., MobileNetV2):  
  - Freeze backbone layers  
  - Add classification head: GlobalAveragePooling2D → Dense → Softmax  
  - Train only head initially, then fine-tune last few layers  
  - Evaluate accuracy, loss, and confusion matrix  

- Compare with baseline ML model results  

---

## 13. Future Work

- Week 4: Deploy the trained deep learning model for real-time plant disease classification  
- Implement federated learning to enable decentralized model updates while preserving data privacy  
- Apply further hyperparameter tuning and model optimization for deployment efficiency  
- Explore additional data augmentation and advanced architectures to improve accuracy  
- Compare performance between centralized and federated deployment approaches  

