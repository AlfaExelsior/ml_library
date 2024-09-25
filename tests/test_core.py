import unittest
import numpy as np
from ml_library.models.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    """Test case for the Linear Regression model."""

    def test_linear_regression(self):
        """Test simple linear regression on a basic dataset."""
        # Simple dataset
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([2, 4, 6])
        
        # Initialize model with learning rate and number of iterations
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        
        # Train model
        model.fit(X, y)
        
        # Predictions
        predictions = model.predict(X)
        
        # Assert that predictions are close to actual y values
        self.assertTrue(np.allclose(predictions, y, rtol=0.1))

    def test_single_feature(self):
        """Test linear regression with a single feature."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Check that predictions match within tolerance
        self.assertTrue(np.allclose(predictions, y, rtol=0.1))

    def test_empty_dataset(self):
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        X = np.array([]).reshape(0, 1)
        y = np.array([])

        with self.assertRaises(ValueError):
            model.fit(X, y)

if __name__ == "__main__":
    unittest.main()
