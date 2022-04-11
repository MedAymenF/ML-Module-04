import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.array,\
 with a for-loop.
Args:
    theta: has to be a numpy.array, a vector of shape n’ * 1.
Return:
    The L2 regularization as a float.
    None if theta in an empty numpy.array.
    None if theta is not of the expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(theta, np.ndarray) or theta.ndim != 2\
            or theta.shape[1] != 1\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be a numpy.array, a vector of shape n’ * 1.")
        return None
    l2 = 0
    for i in range(1, theta.shape[0]):
        l2 += theta[i, 0] ** 2
    return l2


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.array,\
 without any for-loop.
Args:
    theta: has to be a numpy.array, a vector of shape n’ * 1.
Return:
    The L2 regularization as a float.
    None if theta in an empty numpy.array.
    None if theta is not of the expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(theta, np.ndarray) or theta.ndim != 2\
            or theta.shape[1] != 1\
            or not theta.size or not np.issubdtype(theta.dtype, np.number):
        print("theta has to be a numpy.array, a vector of shape n’ * 1.")
        return None
    theta_prime = theta.copy()
    theta_prime[0, 0] = 0
    l2 = theta_prime.T.dot(theta_prime)
    return l2.item()
