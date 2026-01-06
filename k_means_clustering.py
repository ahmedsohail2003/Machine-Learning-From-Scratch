import csv
import numpy as np
import matplotlib.pyplot as plt

# Helper function to load data from CSV
def load_data(filename):
    """
    Load customer profile data from CSV file
    """
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip header if present
        try:
            first_row = next(csv_reader)
            # Check if first row contains headers
            if not all(item.replace('.', '', 1).isdigit() for item in first_row):
                pass  # Skip header
            else:
                # First row contains data, add it back
                data.append([float(item) for item in first_row])
        except StopIteration:
            pass
            
        # Read the rest of the data
        for row in csv_reader:
            data.append([float(row[0]), float(row[1])])
    
    return data

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_means(data, k):
    """
    Implementation of K-means clustering algorithm
    
    Parameters:
        data: array of data points, each containing [x1, x2]
        k: number of clusters
        
    Returns:
        cluster_assignments: list of cluster labels for each data point
        centroids: final centroid coordinates for each cluster
    """
    # Convert data to numpy array for easier computation
    data = np.array(data)
    n = len(data)
    
    # Step 1: Select the first k data points as initial centroids
    centroids = data[:k].copy()
    
    # Initialize variables
    prev_centroids = None
    cluster_assignments = [0] * n
    iteration = 0
    max_iterations = 100  # Prevent infinite loops
    
    # Continue until centroids don't change or max iterations reached
    while prev_centroids is None or not np.array_equal(centroids, prev_centroids):
        if iteration >= max_iterations:
            print("Maximum iterations reached")
            break
            
        # Store current centroids for convergence check
        prev_centroids = centroids.copy()
        
        # Step 2: Assign each data point to the closest centroid based on Euclidean distance
        for i in range(n):
            # Calculate distances to each centroid
            distances = [euclidean_distance(data[i], centroid) for centroid in centroids]
            # Step 3: Label each data point with its respective cluster
            cluster_assignments[i] = np.argmin(distances)
        
        # Step 4: Compute new centroids for each cluster by averaging the data points
        for j in range(k):
            # Get all points assigned to this cluster
            cluster_points = data[np.array(cluster_assignments) == j]
            
            # If the cluster is not empty, update its centroid
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)
        
        iteration += 1
    
    print(f"K-means converged after {iteration} iterations")
    return cluster_assignments, centroids

def plot_data(data, title):
    """
    Plot the data points
    """
    data_array = np.array(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(data_array[:, 0], data_array[:, 1], color='blue', edgecolor='black', s=50)
    plt.title(title, fontsize=14)
    plt.xlabel('Average Monthly Spending (x1)', fontsize=12)
    plt.ylabel('Total Number of Purchases (x2)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

def plot_clusters(data, cluster_assignments, centroids, title):
    """
    Plot the clustered data
    """
    data_array = np.array(data)
    plt.figure(figsize=(10, 6))
    
    # Define colors for each cluster
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot each cluster with a different color
    for i in range(len(centroids)):
        # Get points in this cluster
        cluster_points = data_array[np.array(cluster_assignments) == i]
        
        # Plot the points
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            color=colors[i % len(colors)], 
            edgecolor='black',
            s=50, 
            label=f'Cluster {i+1}'
        )
    
    # Plot the centroids
    plt.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        color='black',
        marker='X', 
        s=200, 
        label='Centroids'
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel('Average Monthly Spending (x1)', fontsize=12)
    plt.ylabel('Total Number of Purchases (x2)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    return plt

def main():
    # Load customer profile data
    customer_data = load_data('CustomerProfiles_Q2.csv')
    print(f"Total number of customers in the dataset: {len(customer_data)}")
    
    # Part A: Plot the customer data points before clustering
    plt_a = plot_data(customer_data, 'Customer Profiles Before Clustering')
    plt_a.savefig('customer_profiles_before_clustering.png')
    plt_a.show()
    
    # Part B: K-means clustering with k=2
    k = 2
    cluster_assignments, centroids = k_means(customer_data, k)
    
    # Print results for parts D and E
    print("\nPart D: Number of data points in each cluster:")
    for i in range(k):
        count = cluster_assignments.count(i)
        print(f"Cluster {i+1}: {count} points")
    
    print("\nPart E: Final centroid coordinates:")
    for i in range(k):
        print(f"Centroid {i+1}: ({centroids[i][0]:.2f}, {centroids[i][1]:.2f})")
    
    # Part C: Plot the final clusters
    plt_c = plot_clusters(customer_data, cluster_assignments, centroids, 'Customer Profiles After Clustering (k=2)')
    plt_c.savefig('customer_profiles_after_clustering.png')
    plt_c.show()

if __name__ == "__main__":
    main()