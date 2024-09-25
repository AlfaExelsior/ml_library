import unittest
import numpy as np
from ml_library.models.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([2, 4, 6])
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertTrue(np.allclose(predictions, y, rtol=0.1))

if __name__ == "__main__":
    unittest.main()
