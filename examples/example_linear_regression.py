import numpy as np
from ml_library.models.linear_regression import LinearRegression
from ml_library.utils import train_test_split

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Output predictions and actual values
print(f"Predicted values: {y_pred}")
print(f"Actual values: {y_test}")
