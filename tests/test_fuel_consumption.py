import unittest
import numpy as np
from ml_library.models.linear_regression import LinearRegression

def standardize(X):
    """Standardize the dataset X (mean 0, variance 1)."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def min_max_scale(X):
    """Min-Max Normalize the dataset X to range [0, 1]."""
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def generate_polynomial_features(X, max_degree=7):
    """Generate polynomial features up to the specified degree."""
    poly_features = X
    for degree in range(2, max_degree + 1):
        poly_features = np.hstack((poly_features, X ** degree))
    return poly_features

class TestFuelConsumptionPrediction(unittest.TestCase):
    """Test case for predicting fuel consumption based on vehicle speed."""
    
    def test_fuel_consumption(self):
        # Speed (km/h) and fuel consumption (L/100km)
        speed = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
        fuel_consumption = np.array([9.5, 8.7, 8.0, 7.5, 7.1, 6.8, 6.5, 6.3, 6.1])

        # Generate polynomial features: speed^2 to speed^7
        speed_features = generate_polynomial_features(speed)

        # Standardize both speed features and fuel consumption
        speed_features_scaled = standardize(speed_features)
        fuel_consumption_scaled = standardize(fuel_consumption)

        # Initialize and train the model with a higher degree polynomial and no regularization
        model = LinearRegression(learning_rate=0.01, n_iterations=15000, l2_penalty=0)
        model.fit(speed_features_scaled, fuel_consumption_scaled)

        # Predict normalized fuel consumption
        predicted_consumption_scaled = model.predict(speed_features_scaled)

        # Reverse the scaling of the predicted values
        predicted_consumption = (predicted_consumption_scaled * np.std(fuel_consumption)) + np.mean(fuel_consumption)

        # Print the actual and predicted values for debugging
        print("Actual fuel consumption:", fuel_consumption)
        print("Predicted fuel consumption:", predicted_consumption)

        # Verify that predicted values are close to actual values (adjusted tolerance)
        self.assertTrue(np.allclose(predicted_consumption, fuel_consumption, atol=0.5))

if __name__ == "__main__":
    unittest.main()
