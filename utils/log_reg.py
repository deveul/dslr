import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import numpy as np

class LogReg():
    def __init__(self, X, y, learning_rate=0.01, iterations=3000, houses=['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'], params={}):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.X = X
        self.y = y
        self.costs = []
        self.houses = houses
        self.params = params

    def predict(self, X):
        return self.sigmoid_function(X.dot(self.params))

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compute_cost(self, params, y):
        m = len(y)
        h = self.sigmoid_function(self.X.dot(params))
        epsilon = 1e-5
        cost = (1/m)*(((-y).T.dot(np.log(h + epsilon)))-((1-y).T.dot(np.log(1-h + epsilon))))
        return cost
    
    def gradient(self, params, y):
        m = len(y)
        y_pred = self.sigmoid_function(self.X.dot(params))
        return  self.X.T.dot((y_pred - y)) / m
    
    def gradient_descent(self):
        for house in self.houses:
            cost_history = np.zeros(self.iterations)
            y = np.where(self.y == house, 1, 0)
            params = np.zeros((self.X.shape[1], 1))
            for i in range(self.iterations):
                params = params - self.learning_rate * self.gradient(params, y)
                cost_history[i] = self.compute_cost(params, y)
            self.params[house] = params.tolist()
            self.costs.append(cost_history)
        return self.params, self.costs