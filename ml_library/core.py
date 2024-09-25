import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between actual and predicted values.
    Parameters:
    y_true : numpy.ndarray : Actual values
    y_pred : numpy.ndarray : Predicted values
    Returns:
    float : Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Compute the R-squared score between actual and predicted values.
    Parameters:
    y_true : numpy.ndarray : Actual values
    y_pred : numpy.ndarray : Predicted values
    Returns:
    float : R-squared score
    """
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)
