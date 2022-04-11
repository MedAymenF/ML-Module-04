#!/usr/bin/env python3
import sys
sys.path.append('../')  # noqa: E402
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ex08.my_logistic_regression import MyLogisticRegression as MyLR
from ex00.polynomial_model_extended import add_polynomial_features
from ex07.benchmark_train import data_splitter
from benchmark_train import one_vs_all, f1_score_


MAX_ITER = 2 * 10 ** 5
ALPHA = 1e-1

# Load the dataset
#  Features
x = pd.read_csv("solar_system_census.csv", index_col=0)
#  Labels
y = pd.read_csv("solar_system_census_planets.csv", index_col=0)
x = x.to_numpy()
y = y.to_numpy()
categories = ["The flying cities of Venus", "United Nations of Earth",
              "Mars Republic", "The Asteroids' Belt colonies"]

# We will use a polynomial hypothesis of degree 3
x_poly = add_polynomial_features(x, 3)

# Normalization
min = x_poly.min(axis=0)
range = x_poly.max(axis=0) - min
x_poly = (x_poly - min) / range

# Split the dataset into a training, a validation, and a test set
(x_rest, x_test, y_rest, y_test) = data_splitter(x_poly, y, 0.8)
(x_train, x_valid, y_train, y_valid) = data_splitter(x_rest, y_rest, 0.8)

# Load models saved from benchmark_train.py
models = pd.read_csv("models.csv").to_numpy()

# Train a model with a polynomial degree of 3
# and a regularization factor of 0.05
print('Training a model with a polynomial degree of 3 and\
 a regularization factor of 0.05:')
best_thetas = one_vs_all(x_train, x_valid, y_train,
                         y_valid, 0.05, categories, evaluate=False)

# Evaluate all the models on the validation set
lambdas = [0, 0.01, 0.05, 0.1, 0.3]
average_f1_scores = []

for i, lambda_ in enumerate(lambdas):
    print('-' * 80)
    print(f'Evaluating a model with a polynomial degree of 3 and\
 a regularization factor of {lambda_:.2f}:\n')
    thetas = models[:, 4 * i: 4 * i + 4]
    all_predictions = []
    for theta in thetas.T:
        my_lr = MyLR(theta.reshape(-1, 1))
        predictions = my_lr.predict_(x_valid)
        all_predictions.append(predictions)
    predictions = np.argmax(np.hstack(all_predictions), axis=1).reshape(-1, 1)
    f1_scores = []
    for zipcode, planet in enumerate(categories):
        f1_score = f1_score_(y_valid, predictions, zipcode)
        print(f'f1 score on {planet} = {f1_score}')
        f1_scores.append(f1_score)
    f1_scores = np.array(f1_scores)
    average_f1_score = f1_scores.mean()
    print(f'Average f1 score: {average_f1_score}')
    average_f1_scores.append(average_f1_score)

# Visualize the performance of the different models with
# a bar plot showing the score
# of the models given their λ value
sns.set(style='darkgrid')
sns.barplot(x=lambdas, y=average_f1_scores)
plt.xlabel('λ')
plt.ylabel('Average f1 score')
plt.show()

# Evaluate the best model on the test set
print('\n\n' + '-' * 80)
print("Evaluating the best model on the test set:")
all_predictions = []
for theta in best_thetas:
    my_lr = MyLR(theta)
    predictions = my_lr.predict_(x_test)
    all_predictions.append(predictions)
predictions = np.argmax(np.hstack(all_predictions), axis=1).reshape(-1, 1)
f1_scores = []
for zipcode, planet in enumerate(categories):
    f1_score = f1_score_(y_test, predictions, zipcode)
    print(f'f1 score on {planet} = {f1_score}')
    f1_scores.append(f1_score)
f1_scores = np.array(f1_scores)
print(f'Average f1 score: {f1_scores.mean()}')

# Calculate the predictions of our best model on the entire dataset
all_predictions = []
for theta in best_thetas:
    my_lr = MyLR(theta)
    predictions = my_lr.predict_(x_poly)
    all_predictions.append(predictions)
all_predictions = np.argmax(np.hstack(all_predictions), axis=1).reshape(-1, 1)
all_correct = all_predictions == y

# Plot 3 scatter plots (one for each pair of citizen features) with the
# dataset and the final prediction of the model
sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y.astype(int).ravel(),
                style=all_correct.ravel(),
                markers={1: 'o',
                0: 'X'}, palette="deep")
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

sns.scatterplot(x=x[:, 0], y=x[:, 2], hue=y.astype(int).ravel(),
                style=all_correct.ravel(),
                markers={1: 'o',
                0: 'X'}, palette="deep")
plt.xlabel('Height')
plt.ylabel('Bone Density')
plt.show()

sns.scatterplot(x=x[:, 1], y=x[:, 2], hue=y.astype(int).ravel(),
                style=all_correct.ravel(),
                markers={1: 'o',
                0: 'X'}, palette="deep")
plt.xlabel('Weight')
plt.ylabel('Bone Density')
plt.show()

# Plot a 3-D scatter plot of the dataset and our predictions
fig = plt.figure()
ax = plt.axes(projection='3d')
correct_predictions = (all_correct == 1).ravel()
colors = y[correct_predictions, :].astype(int)
scatter = ax.scatter(x[correct_predictions, 0],
                     x[correct_predictions, 1],
                     x[correct_predictions, 2],
                     c=colors, cmap="Set1", label='Correct',
                     marker='o')
legend1 = ax.legend(*scatter.legend_elements(), loc="upper left",
                    title="Planets' Zipcodes")
ax.add_artist(legend1)

incorrect_predictions = (all_correct == 0).ravel()
colors = y[incorrect_predictions, :].astype(int)
ax.scatter(x[incorrect_predictions, 0],
           x[incorrect_predictions, 1],
           x[incorrect_predictions, 2],
           c=colors, cmap="Set1", label='Incorrect', marker='X')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Bone Density')
ax.legend()
plt.show()
