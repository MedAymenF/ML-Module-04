import numpy as np
import sys
sys.path.append('../')  # noqa: E402
from ex01.l2_reg import l2


def reg_log_loss_(y, y_hat, theta, lambda_, eps=1e-15):
    """Computes the regularized loss of a logistic regression model\
 from two non-empty numpy.array,
without any for loop. The two arrays must have the same shapes.
Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a vector of shape n * 1.
    lambda_: has to be a float.
    eps: has to be a float, epsilon (default=1e-15).
Return:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.array.
    None if y or y_hat have component ouside [0 ; 1]
    None if y and y_hat do not share the same shapes.
    None if y or y_hat is not of the expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
            or not y.size or not np.issubdtype(y.dtype, np.number):
        print("y has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or\
            y_hat.shape[1] != 1 or not y_hat.size or\
            not np.issubdtype(y_hat.dtype, np.number):
        print("y_hat has to be an numpy.array, a vector of shape m * 1.")
        return None
    if not isinstance(theta, np.ndarray) or theta.ndim != 2\
            or theta.shape[1] != 1\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be a numpy.array, a vector of shape n * 1.")
        return None
    if not isinstance(lambda_, (int, float)):
        print("lambda_ has to be a float.")
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat must have the same shape.')
        return None
    if not isinstance(eps, float):
        print("eps has to be a float.")
        return None
    loss = -(y.T @ np.log(y_hat + eps) +
             (1 - y).T @ np.log(1 - y_hat + eps)) / y.shape[0] +\
        lambda_ * l2(theta) / (2 * y.shape[0])
    return float(loss)
