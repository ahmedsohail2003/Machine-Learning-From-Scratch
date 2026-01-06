# Machine Learning Algorithms From Scratch

This repository contains implementations of fundamental Machine Learning algorithms built entirely from scratch using Python and NumPy. The goal is to demonstrate a deep understanding of the mathematical foundations behind these algorithms without relying on high-level libraries like Scikit-Learn.

## Algorithms Implemented

### 1. k-Nearest Neighbors (k-NN)
**File:** `k_nearest_neighbors.py`
- Implements the k-NN classification algorithm using Euclidean distance.
- Features custom functions for data splitting, accuracy evaluation, and sensitivity analysis across different 'k' values.
- Includes visualization of decision boundaries using Matplotlib.

### 2. K-Means Clustering
**File:** `k_means_clustering.py`
- Implements the K-Means unsupervised learning algorithm.
- Manually handles centroid initialization, distance calculation, and cluster reassignment loops.
- Visualizes the convergence of centroids and customer segmentation groups.

### 3. Perceptron Neural Network
**File:** `perceptron_neural_network.py`
- Implements a single-layer Perceptron for binary classification (Linear Separability).
- Features manual weight updates, bias calculation, and step activation functions.
- achieving 100% convergence on the sample weather dataset.

## Technologies Used
- **Python 3.x**
- **NumPy** (for vector/matrix math)
- **Matplotlib** (for data visualization)
- **Pandas** (for data loading)

## How to Run
1. Clone the repository.
2. Ensure the `.csv` datasets are in the same directory as the scripts.
3. Run any script:
   ```bash
   python k_nearest_neighbors.py
