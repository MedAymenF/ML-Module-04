import numpy as np


def sigmoid_(x):
    """Compute the sigmoid of a vector.
Args:
    x: has to be an numpy.array, a vector
Return:
    The sigmoid value as a numpy.array.
    None otherwise.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or not x.size\
            or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a vector.")
        return None
    return (1 / (1 + np.exp(-x))).reshape(-1, 1)


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three\
 non-empty numpy.array,
with two for-loops. The three arrays must have compatible shapes.
Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    x: has to be a numpy.array, a matrix of dimesion m * n.
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
Return:
    A numpy.array, a vector of shape (n + 1) * 1, containing\
 the results of the formula for all j.
    None if y, x, or theta are empty numpy.array.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1)\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be an numpy.array, a vector of shape (n + 1) * 1.")
        return None
    if not isinstance(lambda_, (int, float)):
        print("lambda_ has to be a float.")
        return None
    if x.shape[0] != y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    gradient = np.zeros(theta.shape)
    m = x.shape[0]
    n = x.shape[1]
    X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    predictions = sigmoid_(X @ theta)
    gradient[0, 0] = (predictions - y).sum() / m
    for j in range(1, n + 1):
        gradient[j, 0] = ((predictions - y).T.dot(x[:, j - 1]) +
                          lambda_ * theta[j, 0]) / m
    return gradient


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three\
 non-empty numpy.array,
without any for-loop. The three arrays must have compatible shapes.
Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    x: has to be a numpy.array, a matrix of dimesion m * n.
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
Return:
    A numpy.array, a vector of shape (n + 1) * 1, containing\
 the results of the formula for all j.
    None if y, x, or theta are empty numpy.array.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1)\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be an numpy.array, a vector of shape (n + 1) * 1.")
        return None
    if not isinstance(lambda_, (int, float)):
        print("lambda_ has to be a float.")
        return None
    if x.shape[0] != y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    m = x.shape[0]
    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    predictions = sigmoid_(X @ theta)
    theta_prime = theta.copy()
    theta_prime[0, 0] = 0
    return (X.T @ (predictions - y) + lambda_ * theta_prime) / m
