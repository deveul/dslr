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

# function to create a list containing mini-batches 
def create_mini_batches(X, y, batch_size): 
    m = y.shape[0]            # number of examples
    n_minibatches = m // batch_size     #number of mini batches
    mini_batches = [] 
    data = np.hstack((X, y))

    # Lets shuffle X and Y
    np.random.shuffle(data) 
    
    i = 0
    for i in range(n_minibatches): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini)) 
    if m % batch_size != 0: 
        last_mini_batch = data[i * batch_size:m]
        X_mini = last_mini_batch[:, :-1] 
        Y_mini = last_mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches

def mini_batch_gradient_descent(categories, X, Y, learning_rate = 0.01, batch_size = 32, iterations = 25): 
    params_dict = {}
    costs = []
    max_iters = iterations
    for category in categories:
        params = np.zeros((X.shape[1], 1)) 
        cost_history = np.zeros(max_iters * (X.shape[0] // batch_size))
        y = np.where(Y == category, 1, 0)
        for i in range(max_iters): 
            mini_batches = create_mini_batches(X, y, batch_size) 
            for j, mini_batch in enumerate(mini_batches): 
                X_mini, y_mini = mini_batch 
                params = params - learning_rate * gradient(params, X_mini, y_mini)
                indice = i * 50 + j
                cost_history[indice] = compute_cost(params, X, y)
        params_dict[category] = params.tolist()
        costs.append(cost_history)
    return params_dict, costs

# def stochastic_gradient_descent(categories, X, Y, learning_rate=0.01, iterations=1):
    # costs = []
    # params = {}
    # for category in categories:
        # cost_history = np.zeros(iterations)
        # y = np.where(Y == category, 1, 0)
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