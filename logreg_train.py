#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import numpy as np

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(X, y, params):
    m = len(y)
    h = sigmoid_function(X.dot(params))
    epsilon = 1e-5
    cost = (1/m)*(((-y).T.dot(np.log(h + epsilon)))-((1-y).T.dot(np.log(1-h + epsilon))))
    return cost

def grad(X, y, params):
    m = len(y)
    return  1/m * X.T.dot((sigmoid_function(X.dot(params)) - y))

def gradient_descent(X, y, params, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        params = params - learning_rate * grad(X, y, params)
        cost_history[i] = compute_cost(X, y, params)
    return params, cost_history

def predict(X, params):
    return np.round(sigmoid_function(X.dot(params)))

def scatter_data(X, y):
    sns.set_style('white')
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y.reshape(-1))
    plt.show()

def plot_cost_history(cost_history):
    plt.figure()
    sns.set_style('white')
    colors = ['r', 'g', 'b', 'gold']
    for i, cost in enumerate(cost_history):
        plt.plot(range(len(cost)), cost, c=colors[i])
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

def plot_result(params_optimal, X, y):
    slope = -(params_optimal[1] / params_optimal[2])
    intercept = -(params_optimal[0] / params_optimal[2])
    
    sns.set_style('white')
    sns.scatterplot(x=X[:,1],y=X[:,2],hue=y.reshape(-1))
    
    ax = plt.gca()
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + (slope * x_vals)
    plt.plot(x_vals, y_vals, c="k")
    plt.show()

class Train:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.coef = {}
        self.cost_history = []

    def get_house(self, house):
        df_house = self.df.copy()
        df_house['Hogwarts House'] = (df_house['Hogwarts House'] == house).astype(int)
        return df_house

    def read_data(self):
        df = pd.read_csv(self.data_file)
        df.dropna(inplace=True)
        self.df = df[['Hogwarts House', 'Astronomy', 'Herbology', 'Ancient Runes']]

    def normalize_values(self):
        pass

    def save_values(self):
        try:
            with open('value_logreg.json', 'w') as json_file:
                json.dump(self.coef, json_file)
                print("value_logreg.json updated")
        except PermissionError:
            print("Vous n'avez pas les droits pour écrire dans le fichier value_lr.json")
            exit()

    def train(self):
        houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
        for house in houses:
            df_house = self.get_house(house)
            X = np.hstack((np.ones((len(df_house), 1)),df_house.drop(columns=['Hogwarts House'])))
            X = (X - np.mean(X)) / np.std(X)
            y = df_house['Hogwarts House']
            initial_thetas = np.zeros(X.shape[1])
            learning_rate = 0.01
            iterations = 3000
            theta, cost_history = gradient_descent(X, y, initial_thetas, learning_rate, iterations)
            self.cost_history.append(cost_history)
            print(theta)
            self.coef[house] = theta.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-c", "--cost_history", help="the csv file containing the data set", action="store_true")
    args = parser.parse_args()
    train = Train(args.data_file)
    train.read_data()
    train.train()
    train.save_values()
    if args.cost_history:
        plot_cost_history(train.cost_history)

if __name__ == "__main__":
    main()