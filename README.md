
# ML Library

A simple machine learning library, focusing on linear regression models. This library includes implementations for linear regression, as well as an example demonstrating fuel consumption predictions using a polynomial regression model.

## Installation

To use the library, you will need to install the required dependencies listed in the `requirements.txt` file. You can install them with:

```bash
pip install -r requirements.txt
```

## Usage

Here is a basic example of how to use the `LinearRegression` model from the library:

```python
from ml_library.models.linear_regression import LinearRegression
import numpy as np

# Example 1: Simple Linear Regression
X = np.array([[1, 1], [2, 2], [3, 3]])
y = np.array([2, 4, 6])

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)
```

### Fuel Consumption Prediction Example

This example demonstrates using polynomial features to predict fuel consumption based on vehicle speed. The model is trained on data of vehicle speed and fuel consumption (L/100km) using a higher-order polynomial regression.

```python
from ml_library.models.linear_regression import LinearRegression
import numpy as np

def standardize(X):
    return (X - np.mean(X)) / np.std(X)

# Speed (km/h) and fuel consumption (L/100km)
speed = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
fuel_consumption = np.array([9.5, 8.7, 8.0, 7.5, 7.1, 6.8, 6.5, 6.3, 6.1])

# Generate polynomial features: speed^2 to speed^7
speed_features = np.hstack([speed**i for i in range(1, 8)])

# Standardize features
speed_features_scaled = standardize(speed_features)
fuel_consumption_scaled = standardize(fuel_consumption)

# Train the linear regression model
model = LinearRegression(learning_rate=0.01, n_iterations=15000)
model.fit(speed_features_scaled, fuel_consumption_scaled)

# Predict fuel consumption
predicted_consumption_scaled = model.predict(speed_features_scaled)
predicted_consumption = (predicted_consumption_scaled * np.std(fuel_consumption)) + np.mean(fuel_consumption)

print("Predicted fuel consumption:", predicted_consumption)
```

## Tests

Unit tests have been included for both basic linear regression and the fuel consumption prediction example. You can run the tests using `unittest`:

```bash
python -m unittest discover -s tests
```

## Contributing

Contributions are welcome! If you encounter issues or have ideas for improvements, feel free to open issues or submit pull requests. Together, we can make this project even better!

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
