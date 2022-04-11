import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns\
 to every power in the range
of 1 up to the power given in argument.
Args:
    x: has to be an numpy.array, where x.shape = (m,n) i.e. a matrix\
 of shape m * n.
    power: has to be a positive integer, the power up to which the columns\
 of matrix x
    are going to be raised.
Return:
    - The matrix of polynomial features as a numpy.array, of shape m * (np),
    containg the polynomial feature values for all training examples.
    - None if x is an empty numpy.array.
    - None if x or power is not of the expected type.
Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or x.ndim != 2\
            or not x.size or not np.issubdtype(x.dtype, np.number):
        print("x has to be an numpy.array, a matrix of shape m * n.")
        return None
    if not isinstance(power, int) or power <= 0:
        print('power has to be a positive integer.')
        return None
    poly = x
    for i in range(2, power + 1):
        poly = np.hstack((poly, x ** i))
    return poly
