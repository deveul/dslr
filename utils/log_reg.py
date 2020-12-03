import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def predict(X, params):
    return sigmoid_function(X.dot(params))

def compute_cost(params, X, y):
    m = len(y)
    h = predict(X, params)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T.dot(np.log(h + epsilon)))-((1-y).T.dot(np.log(1-h + epsilon))))
    return cost

def gradient(params, X, y):
    m = len(y)
    y_pred = predict(X, params)
    return X.T.dot((y_pred - y)) / m

def gradient_descent(categories, X, Y, learning_rate=0.01, iterations=3000):
    costs = []
    params_dict = {}
    for category in categories:
        cost_history = np.zeros(iterations)
        y = np.where(Y == category, 1, 0)
        params = np.zeros((X.shape[1], 1))
        for i in range(iterations):
            params = params - learning_rate * gradient(params, X, y)
            cost_history[i] = compute_cost(params, X, y)
        params_dict[category] = params.flatten().tolist()
        costs.append(cost_history)
    return params_dict, costs

# function to yield mini-batch
def yield_mini_batch(X, y, batch_size):
    length = y.shape[0]
    for i in np.arange(0, length, batch_size):
        yield X[i: i + batch_size], y[i: i + batch_size]

def mini_batch_gradient_descent(categories, X, Y, learning_rate=0.1, batch_size=32, iterations=5): 
    params_dict = {}
    costs = []
    max_iters = iterations
    for category in categories:
        params = np.zeros((X.shape[1], 1)) 
        cost_history = np.zeros(max_iters * (X.shape[0] // batch_size))
        y = np.where(Y == category, 1, 0)
        for i in range(max_iters): 
            j = 0
            for X_mini, y_mini in yield_mini_batch(X, y, batch_size): 
                params = params - learning_rate * gradient(params, X_mini, y_mini)
                indice = i * 50 + j
                j += 1
                cost_history[indice] = compute_cost(params, X, y)
        params_dict[category] = params.flatten().tolist()
        costs.append(cost_history)
    return params_dict, costs