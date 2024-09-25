import torch
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, l2_penalty=0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Handle edge case: empty dataset
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty dataset provided.")
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.l2_penalty * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


