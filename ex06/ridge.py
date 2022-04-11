import numpy as np


class MyLinearRegression():
    """Description:
    My personnal linear regression class to fit like a boss."""

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        error_msg = "thetas has to be an numpy.array or list,\
 a vector."
        if isinstance(thetas, np.ndarray):
            if thetas.ndim != 2 or thetas.shape[1] != 1 or not thetas.size\
                    or not np.issubdtype(thetas.dtype, np.number):
                print(error_msg)
                return None
        elif isinstance(thetas, list):
            try:
                thetas = np.array(thetas).reshape((-1, 1))
                assert np.issubdtype(thetas.dtype, np.number)
            except Exception:
                print(error_msg)
                return None
        else:
            print(error_msg)
            return None
        if not isinstance(alpha, (float, int)):
            print("alpha has to be a float.")
            return None
        if alpha <= 0:
            print("The learning rate has to be strictly positive.")
            return None
        if not isinstance(max_iter, int):
            print("max_iter has to be an int.")
            return None
        if max_iter < 0:
            print("The number of iterations has to be positive.")
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def fit_(self, x, y):
        if not hasattr(self, 'thetas') or not hasattr(self, 'alpha')\
                or not hasattr(self, 'max_iter'):
            return None
        if not isinstance(x, np.ndarray) or x.ndim != 2\
                or not x.size or not np.issubdtype(x.dtype, np.number):
            print("x has to be an numpy.array, a matrix of shape m * n.")
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
                or not y.size or not np.issubdtype(y.dtype, np.number):
            print("y has to be an numpy.array, a vector of shape m * 1.")
            return None
        if x.shape[0] != y.shape[0]:
            print('x and y must have the same number of rows.')
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            print("x and theta's shapes don't match.")
            return None
        X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        for _ in range(self.max_iter):
            grad = X.T @ (X @ self.thetas - y) / x.shape[0]
            self.thetas = self.thetas - self.alpha * grad

    def predict_(self, x):
        if not hasattr(self, 'thetas'):
            return None
        if not isinstance(x, np.ndarray) or x.ndim != 2\
                or not x.size or not np.issubdtype(x.dtype, np.number):
            print("x has to be an numpy.array, a matrix of shape m * n.")
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            print("x and theta's shapes don't match.")
            return None
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return x @ self.thetas

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
                or not y.size or not np.issubdtype(y.dtype, np.number):
            print("y has to be an numpy.array, a vector.")
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or\
                y_hat.shape[1] != 1 or not y_hat.size or\
                not np.issubdtype(y_hat.dtype, np.number):
            print("y_hat has to be an numpy.array, a vector.")
            return None
        if y.shape[0] != y_hat.shape[0]:
            print('y and y_hat have different shapes')
            return None
        return (y - y_hat) ** 2

    def loss_(self, y, y_hat):
        squared_error = self.loss_elem_(y, y_hat)
        if squared_error is None:
            return None
        return float(squared_error.sum() / (2 * y.shape[0]))

    def mse_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
                or not y.size or not np.issubdtype(y.dtype, np.number):
            print("y has to be an numpy.array, a vector.")
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or\
                y_hat.shape[1] != 1 or not y_hat.size or\
                not np.issubdtype(y_hat.dtype, np.number):
            print("y_hat has to be an numpy.array, a vector.")
            return None
        error = y - y_hat
        return float(error.T.dot(error)[0, 0] / y.shape[0])


class MyRidge(MyLinearRegression):
    """Description:
    My personnal ridge regression class to fit like a boss."""

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha, max_iter)
        if not isinstance(lambda_, (float, int)):
            print("lambda_ has to be a float.")
            return None
        if lambda_ < 0:
            print("lambda_ has to be positive.")
            return None
        self.lambda_ = lambda_

    def get_params_(self):
        if not hasattr(self, 'thetas'):
            return None
        return self.thetas

    def set_params_(self, thetas):
        error_msg = "thetas has to be an numpy.array or list,\
 a vector."
        if isinstance(thetas, np.ndarray):
            if thetas.ndim != 2 or thetas.shape[1] != 1 or not thetas.size\
                    or not np.issubdtype(thetas.dtype, np.number):
                print(error_msg)
                return None
        elif isinstance(thetas, list):
            try:
                thetas = np.array(thetas).reshape((-1, 1))
                assert np.issubdtype(thetas.dtype, np.number)
            except Exception:
                print(error_msg)
                return None
        else:
            print(error_msg)
            return None
        self.thetas = thetas

    def loss_(self, y, y_hat):
        if not hasattr(self, 'thetas') or not hasattr(self, 'lambda_'):
            return None
        squared_error = self.loss_elem_(y, y_hat)
        if squared_error is None:
            return None
        reg = self.thetas[1:, :].T.dot(self.thetas[1:, :]).item()
        return float((squared_error.sum() + self.lambda_ * reg) /
                     (2 * y.shape[0]))

    def gradient_(self, x, y):
        if not hasattr(self, 'thetas') or not hasattr(self, 'lambda_'):
            return None
        if not isinstance(x, np.ndarray) or x.ndim != 2\
                or not x.size or not np.issubdtype(x.dtype, np.number):
            print("x has to be an numpy.array, a matrix of shape m * n.")
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
                or not y.size or not np.issubdtype(y.dtype, np.number):
            print("y has to be an numpy.array, a vector of shape m * 1.")
            return None
        if x.shape[0] != y.shape[0]:
            print('x and y must have the same number of rows.')
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            print("x and theta's shapes don't match.")
            return None
        m = x.shape[0]
        X = np.concatenate((np.ones((m, 1)), x), axis=1)
        predictions = X @ self.thetas
        theta_prime = self.thetas.copy()
        theta_prime[0, 0] = 0
        return (X.T @ (predictions - y) + self.lambda_ * theta_prime) / m

    def fit_(self, x, y):
        if not hasattr(self, 'thetas') or not hasattr(self, 'alpha')\
                or not hasattr(self, 'max_iter')\
                or not hasattr(self, 'lambda_'):
            return None
        if not isinstance(x, np.ndarray) or x.ndim != 2\
                or not x.size or not np.issubdtype(x.dtype, np.number):
            print("x has to be an numpy.array, a matrix of shape m * n.")
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1\
                or not y.size or not np.issubdtype(y.dtype, np.number):
            print("y has to be an numpy.array, a vector of shape m * 1.")
            return None
        if x.shape[0] != y.shape[0]:
            print('x and y must have the same number of rows.')
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            print("x and theta's shapes don't match.")
            return None
        m = x.shape[0]
        X = np.concatenate((np.ones((m, 1)), x), axis=1)
        for _ in range(self.max_iter):
            theta_prime = self.thetas.copy()
            theta_prime[0, 0] = 0
            grad = (X.T @ (X @ self.thetas - y) +
                    self.lambda_ * theta_prime) / m
            self.thetas = self.thetas - self.alpha * grad
