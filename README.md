# kaggle-PLANT-ml
WIDS-2025-END-TO-END-ML
# 🌱 PlantVillage Disease Classification – End-to-End Pipeline

## 1. Project Overview

This project focuses on building a complete image analysis and baseline machine learning pipeline for the **PlantVillage plant disease dataset**. The work covers dataset exploration, preprocessing, data integrity checks, statistical analysis, and a classical machine learning baseline using handcrafted features.

The objective is to thoroughly understand the dataset before moving toward advanced deep learning approaches.

---

## 2. Dataset Description

The PlantVillage dataset contains images of healthy and diseased plant leaves across multiple crop species.

The dataset is organized into three image representations:
- `color`
- `grayscale`
- `segmented`

For this project, **only color images are used**, since plant diseases are primarily identified through color-based symptoms such as yellowing, spots, and discoloration. Using only one representation also prevents data duplication and leakage.

---

## 3. Dataset Traversal and Metadata Creation

The dataset directory structure is programmatically traversed to collect metadata for each image.

For every image, the following information is extracted:
- Image type (color / grayscale / segmented)
- Disease class (derived from the folder name)
- Image filename

This information is stored in a pandas DataFrame to enable structured analysis and easy inspection.

---

## 4. Image Path Construction

A full absolute image path is constructed for every image by combining:
- Dataset root directory
- Image type directory
- Class directory
- Image filename

Storing absolute paths ensures reliable image loading during preprocessing and feature extraction.

---

## 5. Class Distribution Analysis

The number of images in each disease class is computed using value counts.

### Key observations:
- The dataset is **highly imbalanced**
- Some disease classes contain more than **15,000 images**
- Other classes contain fewer than **500 images**
- Tomato and citrus diseases dominate the dataset

This imbalance is an important factor to consider during model training.

---

## 6. Visualization of Class Distribution

The class distribution is visualized using:
- Histogram plots for clear comparison
- Pie charts to understand proportional representation

These visualizations help identify dominant and minority classes quickly.

---

## 7. Image Size Analysis

The height and width of each image are extracted using the PIL library and stored in the DataFrame.

### Observation:
- All images have a resolution of **256 × 256 pixels**

Since all images share the same dimensions, image size scatter plots collapse into a **single point**, confirming dataset consistency.

---

## 8. Corrupted Image Detection

Each image file is checked for corruption by attempting to open it.

### Process:
- Every image path is loaded
- Unreadable or broken files are flagged
- A progress bar is used to track execution

### Result:
- **0 corrupted images found**
- The dataset is clean and suitable for training

---

## 9. Class Imbalance Considerations

The dataset exhibits significant class imbalance.

To handle this:
- Stratified train–test splitting is used
- Class distribution is preserved during evaluation

More advanced strategies such as class weighting or resampling can be applied in future iterations.

---

## 10. Project Workflow (Baseline Machine Learning Pipeline)

The pipeline begins by selecting only color images from the dataset to avoid redundancy and leakage.

The dataset is traversed to collect class labels and image paths, which are stored in a pandas DataFrame.

Disease class names are converted into numerical labels using label encoding to make them compatible with machine learning models.

Before feature extraction, all images are validated to ensure there are no corrupted files.

Feature extraction is performed using **Histogram of Oriented Gradients (HOG)**. Each image is resized to a fixed resolution, converted to grayscale, and processed to extract gradient-based features that capture structural and texture information relevant to plant diseases.

The dataset is split into training and testing sets using a **stratified split**, with 80% of the data used for training and 20% for evaluation.

Feature scaling is applied using standardization to ensure all features have zero mean and unit variance, which is crucial for distance-based models.

To reduce dimensionality and improve efficiency, **Principal Component Analysis (PCA)** is applied, retaining 95% of the original variance.

Finally, a shallow machine learning model is trained on the processed features. Performance is evaluated using accuracy and a confusion matrix to understand class-wise behavior.

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
- Baseline machine learning model training

---

## 12. Future Work

- Implement deep learning models (CNNs)
- Apply data augmentation to address imbalance
- Perform hyperparameter tuning
- Compare classical ML and deep learning performance

---

This project establishes a strong and clean foundation for plant disease classification using computer vision techniques.
