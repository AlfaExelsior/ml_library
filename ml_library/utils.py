import numpy as np


def normalize(X):
    """
    Normalize the dataset X to have mean 0 and standard deviation 1.
    Parameters:
    X : numpy.ndarray : Dataset
    Returns:
    numpy.ndarray : Normalized dataset
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def train_test_split(X, y, test_size=0.2):
    """
    Split the dataset into training and testing sets.
    Parameters:
    X : numpy.ndarray : Features
    y : numpy.ndarray : Labels
    test_size : float : Proportion of the dataset to include in the test split
    Returns:
    tuple : Training and testing data for both X and y
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(X.shape[0] * (1 - test_size))
    X_train = X[indices[:split]]
    X_test = X[indices[split:]]
    y_train = y[indices[:split]]
    y_test = y[indices[split:]]
    return X_train, X_test, y_train, y_test
