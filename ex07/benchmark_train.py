#!/usr/bin/env python3
import sys
sys.path.append('../')  # noqa: E402
import pandas as pd
import numpy as np
from ex06.ridge import MyRidge as MyR
from ex00.polynomial_model_extended import add_polynomial_features
from numpy.random import default_rng


def data_splitter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)\
 into a training and a test set,
while respecting the given proportion of examples to be kept\
 in the training set.
Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    proportion: has to be a float, the proportion of the dataset\
 that will be assigned to the
    training set.
Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible shapes.
    None if x, y or proportion is not of expected type.
Raises:
    This function should not raise any Exception.
"""
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
    if not isinstance(proportion, (int, float)):
        print('proportion has to be a float.')
        return None
    if proportion < 0 or proportion > 1:
        print('proportion has to be between 0 and 1.')
        return None
    rng = default_rng(1337)
    z = np.hstack((x, y))
    rng.shuffle(z)
    x, y = z[:, :-1].reshape(x.shape), z[:, -1].reshape(y.shape)
    idx = int((x.shape[0] * proportion))
    x_train, x_test = np.split(x, [idx])
    y_train, y_test = np.split(y, [idx])
    return (x_train, x_test, y_train, y_test)


MAX_ITER = 10 ** 6
ALPHA = 1e-1


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("space_avocado.csv", index_col=0)
    x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
    n_features = x.shape[1]
    y = np.array(df['target']).reshape(-1, 1)

    # Split the dataset into a training, a validation, and a test set
    (x_rest, x_test, y_rest, y_test) = data_splitter(x, y, 0.8)
    (x_train, x_valid, y_train, y_valid) = data_splitter(x_rest, y_rest, 0.8)

    # Train four separate Linear Regression models with polynomial hypothesis
    # with degrees ranging from 1 to 4
    # For each hypothesis consider a regularized factor ranging from 0 to 1
    # with a step of 0.2
    degrees = list(range(1, 5))
    lambdas = np.arange(0, 1, 0.2).tolist()
    models = []

    for degree in degrees:
        print('-' * 80 * (degree != 1))
        mse_list = []
        models_list = []
        for lambda_ in lambdas:
            print(f'Training a model with polynomial degree\
 {degree} and a regularization factor of {lambda_:.1f}:\n')
            my_lr = MyR(np.ones((1 + n_features * degree, 1)),
                        ALPHA, MAX_ITER, lambda_)
            x_poly = add_polynomial_features(x_train, degree)

            # Normalization
            min = x_poly.min(axis=0)
            range = x_poly.max(axis=0) - min
            x_poly = (x_poly - min) / range

            my_lr.fit_(x_poly, y_train)
            predictions = my_lr.predict_(x_poly)
            train_mse = my_lr.mse_(y_train, predictions)
            print(f'Training set MSE = {train_mse:e}')

            # Evaluate model on the validation set
            x_poly = add_polynomial_features(x_valid, degree)

            # Normalization
            x_poly = (x_poly - min) / range

            predictions = my_lr.predict_(x_poly)
            valid_mse = my_lr.mse_(y_valid, predictions)
            print(f'Validation set MSE = {valid_mse:e}\n\n')

            mse_list.append(valid_mse)
            models_list.append(my_lr)

            # Save model parameters to a list
            # We need them in space_avocado.py
            thetas = my_lr.get_params_().copy()
            thetas.resize((1 + 3 * 4, 1))
            # print(repr(thetas))
            models.append(thetas)

        # Select the best λ value for this hypothesis
        mse_list = np.array(mse_list)
        idx = np.argmin(mse_list)
        best_lambda_ = lambdas[idx]
        best_model = models_list[idx]
        print(f"The best value of λ for the model with\
 polynomial degree {degree} is {best_lambda_}.\n")

        # Evaluate the best model on the test set
        x_poly = add_polynomial_features(x_test, degree)
        # Normalization
        x_poly = (x_poly - min) / range
        predictions = best_model.predict_(x_poly)
        test_mse = best_model.mse_(y_test, predictions)
        print(f'Test set MSE = {test_mse:e}')

    # Save the parameters of the different models into a file
    all_models = np.hstack(models)
    pd.DataFrame(all_models).to_csv("models.csv",
                                    float_format='%010.1f', index=False)
