# Machine Learning Algorithms Implementation from Scratch

## Overview
This repository contains implementations of fundamental Machine Learning algorithms built entirely from scratch using **Python** and **NumPy**. 

The primary objective of this project is to demonstrate a deep understanding of the mathematical underpinnings of these algorithms (Linear Algebra, Euclidean Geometry, Vector Calculus) by implementing them without relying on high-level abstractions like Scikit-Learn.

## Contents

### 1. k-Nearest Neighbors (k-NN) Classification
**File:** `k_nearest_neighbors.py`
**Dataset:** `CustomerDataset_Q1.csv`
*   **Implementation:** Developed a custom k-NN classifier using Euclidean distance calculations.
*   **Features:** 
    *   Manual implementation of majority voting logic.
    *   Sensitivity analysis comparing performance across $k=1$ to $k=4$.
    *   Impact analysis of varying training set splits (80/20 vs 50/50).

### 2. K-Means Clustering (Unsupervised Learning)
**File:** `k_means_clustering.py`
**Dataset:** `CustomerProfiles_Q2.csv`
*   **Implementation:** Built the K-Means algorithm to segment customer profiles based on spending habits.
*   **Features:**
    *   Random centroid initialization.
    *   Iterative convergence loop based on minimizing intra-cluster variance.
    *   Visualization of cluster separation.

### 3. Perceptron Neural Network
**File:** `perceptron_neural_network.py`
**Dataset:** `WeatherData_Q3.csv`
*   **Implementation:** Constructed a single-layer Perceptron for binary classification (Linear Separability).
*   **Features:**
    *   Manual coding of weight updates and bias terms.
    *   Step activation function implementation.
    *   Achieved 100% convergence on linearly separable weather data.

---

## Technical Analysis Report
For a detailed breakdown of the results, including decision boundary visualizations, convergence graphs, and an additional analysis on **Constraint Satisfaction Problems (AC-3 Algorithm)**, please refer to the technical report included in this repository:

📄 **[View Technical Analysis Report (PDF)](Technical_Analysis_Report.pdf)**

---

## Dependencies
*   **Python 3.x**
*   **NumPy:** For vector and matrix operations.
*   **Matplotlib:** For data visualization.
*   **Pandas:** For CSV data loading.

## How to Run
Ensure the `.csv` datasets are located in the same directory as the scripts. You can run each algorithm individually via the terminal:

```bash
# Run k-NN Classification
python k_nearest_neighbors.py

# Run K-Means Clustering
python k_means_clustering.py

# Run Perceptron Neural Network
python perceptron_neural_network.py
