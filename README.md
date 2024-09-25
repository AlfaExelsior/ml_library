
# ML Library

A simple machine learning library for beginners.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from ml_library.models.linear_regression import LinearRegression
import numpy as np

X = np.array([[1, 1], [2, 2], [3, 3]])
y = np.array([2, 4, 6])

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print(predictions)
```

## Contributing

Feel free to submit issues or pull requests if you want to contribute to the development of this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
