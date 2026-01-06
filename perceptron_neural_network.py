import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load the weather data
weather_data = pd.read_csv('WeatherData_Q3.csv')

# PROBLEM 3A: Visualize the Weather Data

# Separate data into Rain and No Rain groups
rain_data = weather_data[weather_data['rain'] == 1]
no_rain_data = weather_data[weather_data['rain'] == 0]

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Plot No Rain points as blue squares
plt.scatter(no_rain_data['temp'], no_rain_data['humid'], 
            color='blue', marker='s', s=100, label='No Rain (y=0)')

# Plot Rain points as red circles
plt.scatter(rain_data['temp'], rain_data['humid'], 
            color='red', marker='o', s=100, label='Rain (y=1)')

# Add labels and title
plt.xlabel('Temperature (scaled)')
plt.ylabel('Humidity (scaled)')
plt.title('Weather Data: Temperature vs. Humidity for Rain Prediction')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# PROBLEM 3B: Implement a Perceptron Model

# Split the dataset into training (first 15 instances) and test set (last 5 instances)
train_data = weather_data.iloc[:15]
test_data = weather_data.iloc[15:]

# Extract features (temperature and humidity) and labels (rain)
X_train = train_data[['temp', 'humid']].values
y_train = train_data['rain'].values
X_test = test_data[['temp', 'humid']].values
y_test = test_data['rain'].values

class Perceptron:
    def __init__(self, learning_rate=0.1, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def initialize_weights(self, n_features):
        # Initialize weights and bias randomly between -0.5 and 0.5
        self.weights = np.random.uniform(-0.5, 0.5, n_features)
        self.bias = np.random.uniform(-0.5, 0.5)
        
    def activation_function(self, x):
        # Step function
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        # Calculate net input: X · weights + bias
        net_input = np.dot(X, self.weights) + self.bias
        # Apply activation function
        return self.activation_function(net_input)
    
    def predict_batch(self, X):
        # Make predictions for multiple samples
        predictions = []
        for x in X:
            predictions.append(self.predict(x))
        return np.array(predictions)
    
    def train(self, X, y):
        # Initialize weights
        n_features = X.shape[1]
        self.initialize_weights(n_features)
        
        # Track errors during training
        errors = []
        
        # Training iterations
        for iteration in range(self.max_iterations):
            error_count = 0
            
            # Iterate through each training sample
            for x, target in zip(X, y):
                # Make prediction
                prediction = self.predict(x)
                
                # Update weights and bias if prediction is incorrect
                if prediction != target:
                    # Calculate error
                    error = target - prediction
                    
                    # Update weights
                    self.weights += self.learning_rate * error * x
                    
                    # Update bias
                    self.bias += self.learning_rate * error
                    
                    # Increment error count
                    error_count += 1
            
            # Record errors for this iteration
            errors.append(error_count)
            
            # If no errors, perceptron has converged
            if error_count == 0:
                print(f"Perceptron converged after {iteration + 1} iterations")
                break
                
        return errors
    
    def calculate_accuracy(self, X, y):
        predictions = self.predict_batch(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy

# Create and train the perceptron
perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
errors = perceptron.train(X_train, y_train)

# Calculate and report accuracies
train_accuracy = perceptron.calculate_accuracy(X_train, y_train)
test_accuracy = perceptron.calculate_accuracy(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot training errors over iterations (if the perceptron didn't converge immediately)
if len(errors) > 1:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Misclassifications')
    plt.title('Perceptron Training: Misclassifications per Iteration')
    plt.grid(True)
    plt.show()

# Display the final weights and bias
print(f"Final weights: {perceptron.weights}")
print(f"Final bias: {perceptron.bias}")