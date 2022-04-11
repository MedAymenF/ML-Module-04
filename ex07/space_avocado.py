#!/usr/bin/env python3
import sys
sys.path.append('../')  # noqa: E402
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ex06.ridge import MyRidge as MyR
from benchmark_train import data_splitter
from ex00.polynomial_model_extended import add_polynomial_features


# Load the dataset
df = pd.read_csv("space_avocado.csv", index_col=0)
x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
n_features = x.shape[1]
y = np.array(df['target']).reshape(-1, 1)


# Split the dataset into a training set and a test set
(x_train, x_test, y_train, y_test) = data_splitter(x, y, 0.8)


# Train a model with a polynomial degree of 4 and a regularization factor of 0
print('Training a model with a polynomial degree of 4 and\
 a regularization factor of 0.')
degree = 4
my_lr = MyR(np.ones((1 + n_features * degree, 1)), 1e-1, 10 ** 6, 0)
x_poly = add_polynomial_features(x_train, degree)


# Normalization
min = x_poly.min(axis=0)
rnge = x_poly.max(axis=0) - min
x_poly = (x_poly - min) / rnge

my_lr.fit_(x_poly, y_train)
train_predictions = my_lr.predict_(x_poly)
train_mse = my_lr.mse_(y_train, train_predictions)
print(f'Training set MSE = {train_mse:e}')


# Evaluate model on the test set
x_poly = add_polynomial_features(x_test, degree)
x_poly = (x_poly - min) / rnge
test_predictions = my_lr.predict_(x_poly)
test_mse = my_lr.mse_(y_test, test_predictions)
print(f'Test set MSE = {test_mse:e}\n\n')


# Load models saved from benchmark_train.py
models = pd.read_csv("models.csv").to_numpy()


# Evaluate all the models we've trained in benchmark_train.py
degrees = list(range(1, 5))
lambdas = np.arange(0, 1, 0.2).tolist()

evaluation_data = []
for degree in degrees:
    print('-' * 80 * (degree != 1))
    for i, lambda_ in enumerate(lambdas):
        print(f'Evaluating a model with polynomial degree {degree} and\
 a regularization factor of {lambda_:.1f}:\n')
        n_rows = 1 + n_features * degree
        col = (degree - 1) * len(lambdas) + i
        saved_theta = models[:n_rows, col].copy().reshape(-1, 1)
        my_lr = MyR(saved_theta)
        # Evaluate model on the test set
        x_poly = add_polynomial_features(x_test, degree)

        # Normalization
        min = x_poly.min(axis=0)
        range = x_poly.max(axis=0) - min
        x_poly = (x_poly - min) / range

        predictions = my_lr.predict_(x_poly)
        test_mse = my_lr.mse_(y_test, predictions)
        print(f'Test set MSE = {test_mse:e}\n\n')

        evaluation_data.append([degree, lambda_, test_mse])


# Plot a bar plot showing the MSE score of the models in function of the
# polynomial degree of the hypothesis and the regularization factor
data_df = pd.DataFrame(evaluation_data,
                       columns=['Degree', 'Lambda',
                                'Test Set Mean Squared Error'])
sns.set_style("darkgrid")
sns.catplot(data=data_df, x='Lambda', y='Test Set Mean Squared Error',
            col='Degree', kind='bar')
plt.show()


# Plot the true price and the predicted price obtained via
# the best model (degree 4) for each value of λ
thetas = models[:, -5:]
# Calculate the predictions for each value of λ
all_predictions = []
for (theta, lambda_) in zip(thetas.T, lambdas):
    my_lr = MyR(theta.reshape(-1, 1), lambda_=lambda_)
    x_poly = add_polynomial_features(x, 4)

    # Normalization
    min = x_poly.min(axis=0)
    range = x_poly.max(axis=0) - min
    x_poly = (x_poly - min) / range

    predictions = my_lr.predict_(x_poly)
    all_predictions.append(predictions)

# Plot `weight` on the x axis
plt.scatter(x[:, 0], y, label='True values')
for (predictions, lambda_) in zip(all_predictions, lambdas):
    plt.scatter(x[:, 0], predictions,
                label=f'Predictions for λ = {lambda_:.2f}')
plt.xlabel('weight (in ton)')
plt.ylabel('target (in trantorian unit)')
plt.legend()
plt.show()

# Plot `prod_distance` on the x axis
plt.scatter(x[:, 1], y, label='True values')
for (predictions, lambda_) in zip(all_predictions, lambdas):
    plt.scatter(x[:, 1], predictions,
                label=f'Predictions for λ = {lambda_:.2f}')
plt.xlabel('prod_distance (in Mkm)')
plt.ylabel('target (in trantorian unit)')
plt.legend()
plt.show()


# Plot `time_delivery` on the x axis
plt.scatter(x[:, 2], y, label='True values')
for (predictions, lambda_) in zip(all_predictions, lambdas):
    plt.scatter(x[:, 2], predictions,
                label=f'Predictions for λ = {lambda_:.2f}')
plt.xlabel('time_delivery (in days)')
plt.ylabel('target (in trantorian unit)')
plt.legend()
plt.show()
