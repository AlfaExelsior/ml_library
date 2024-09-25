import numpy as np
import matplotlib.pyplot as plt
from ml_library.models.linear_regression import LinearRegression

def normalize(X):
    """Normalize the dataset X to have mean 0 and standard deviation 1."""
    return (X - np.mean(X)) / np.std(X)

# Speed (km/h) and Fuel consumption (L/100km)
speed = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
fuel_consumption = np.array([9.5, 8.7, 8.0, 7.5, 7.1, 6.8, 6.5, 6.3, 6.1])

# Generate polynomial features: add speed^2 as a feature
speed_squared = speed ** 2
speed_features = np.hstack((speed, speed_squared))  # Combine original speed and speed^2

# Normalize both speed features and fuel consumption
speed_features_normalized = normalize(speed_features)
fuel_consumption_normalized = normalize(fuel_consumption)

# Initialize and train the linear regression model
model = LinearRegression(learning_rate=0.0005, n_iterations=3000)
model.fit(speed_features_normalized, fuel_consumption_normalized)

# Predict normalized fuel consumption based on normalized speed features
predicted_consumption_normalized = model.predict(speed_features_normalized)

# Reverse the normalization of the predicted values
predicted_consumption = (predicted_consumption_normalized * np.std(fuel_consumption)) + np.mean(fuel_consumption)

# Visualize actual data vs predicted results
plt.scatter(speed, fuel_consumption, color="blue", label="Actual Consumption")
plt.plot(speed, predicted_consumption, color="red", label="Predicted Consumption")
plt.xlabel('Speed (km/h)')
plt.ylabel('Fuel Consumption (L/100km)')
plt.title('Fuel Consumption vs Speed (With Polynomial Features)')
plt.legend()
plt.show()

# Display predicted results
print("Predicted fuel consumption (L/100km):", predicted_consumption)

