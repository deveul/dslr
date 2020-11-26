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
        params_dict[category] = params.tolist()
        costs.append(cost_history)
    return params_dict, costs

# def stochastic_gradient_descent(categories, X, yi, learning_rate=0.01, iterations=1):
    # costs = []
    # params = {}
    # for category in categories:
        # cost_history = np.zeros(iterations)
        # y = np.where(yi == category, 1, 0)
        # params = np.zeros((X.shape[1], 1))
        # m = len(y)
        # for _ in range(iterations):
            # for i in range(m):
                # rand_ind = np.random.randint(0,m)
                # X_i = X[rand_ind,:].reshape(1,X.shape[1])
                # y_i = y[rand_ind].reshape(1,1)
                # prediction = np.dot(X_i, params)
# 
                # params = params - learning_rate * X_i.T.dot((prediction - y_i)) / m
                # cost_history[i] = compute_cost(params, X_i, y_i)
            # params[category] = params.tolist()
            # costs.append(cost_history)
    # return params, costs