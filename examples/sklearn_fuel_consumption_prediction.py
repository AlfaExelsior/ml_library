from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    """Normalize the dataset X to have mean 0 and standard deviation 1."""
    return (X - np.mean(X)) / np.std(X)

# Speed (km/h) and Fuel consumption (L/100km)
speed = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
fuel_consumption = np.array([9.5, 8.7, 8.0, 7.5, 7.1, 6.8, 6.5, 6.3, 6.1])

# Generate polynomial features: add speed^2 as a feature
speed_squared = speed ** 2
speed_features = np.hstack((speed, speed_squared))

# Normalize both speed features and fuel consumption
speed_features_normalized = normalize(speed_features)
fuel_consumption_normalized = normalize(fuel_consumption)

# Use sklearn's linear regression for comparison
sklearn_model = SklearnLinearRegression()
sklearn_model.fit(speed_features_normalized, fuel_consumption_normalized)

# Predict using sklearn model
predicted_consumption_normalized = sklearn_model.predict(speed_features_normalized)

# Reverse the normalization of the predicted values
predicted_consumption = (predicted_consumption_normalized * np.std(fuel_consumption)) + np.mean(fuel_consumption)

# Visualize actual data vs predicted results
plt.scatter(speed, fuel_consumption, color="blue", label="Actual Consumption")
plt.plot(speed, predicted_consumption, color="red", label="Predicted Consumption (sklearn)")
plt.xlabel('Speed (km/h)')
plt.ylabel('Fuel Consumption (L/100km)')
plt.title('Fuel Consumption vs Speed (With Polynomial Features using sklearn)')
plt.legend()
plt.show()

# Display predicted results
print("Predicted fuel consumption (L/100km) using sklearn:", predicted_consumption)
