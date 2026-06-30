import random

import numpy as np

from perceptron_neural_network import Perceptron


def test_activation_function_step():
    perceptron = Perceptron()
    # Step function: >= 0 maps to 1, negatives map to 0.
    assert perceptron.activation_function(0) == 1
    assert perceptron.activation_function(-1) == 0


def test_train_learns_and_dataset():
    np.random.seed(42)
    random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    perceptron.train(X, y)

    # AND is linearly separable, so the perceptron reaches perfect accuracy.
    assert perceptron.calculate_accuracy(X, y) == 1.0


def test_predict_returns_int_label():
    np.random.seed(42)
    random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    perceptron.train(X, y)

    prediction = perceptron.predict(np.array([1, 1]))
    assert isinstance(prediction, int)
    assert prediction in (0, 1)
