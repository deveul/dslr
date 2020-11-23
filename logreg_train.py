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
from utils.visuals import plot_cost_history
from utils.log_reg import LogReg
from utils.stats_functions import dslr_mean
from utils.stats_functions import dslr_std

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

class Train:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.reg = None

    def get_house(self, house):
        df_house = self.df.copy()
        df_house['Hogwarts House'] = (df_house['Hogwarts House'] == house).astype(int)
        return df_house

    def read_data(self):
        df = pd.read_csv(self.data_file, usecols=['Hogwarts House', 'Astronomy', 'Herbology', 'Ancient Runes'])
        for column in df:
            if is_numeric_dtype(df[column].dtypes):
                df[column] = df[column].fillna(dslr_mean(df[column].dropna()))
                df[column] = self.standardize_values(df[column])
        self.df = df

    def standardize_values(self, X):
        X = (X - dslr_mean(X)) / dslr_std(X)
        return X

    def save_values(self, params):
        try:
            with open('value_logreg.json', 'w') as json_file:
                json.dump(params, json_file)
                print("value_logreg.json updated")
        except PermissionError:
            print("Vous n'avez pas les droits pour écrire dans le fichier value_lr.json")
            exit()

    def train(self):
        houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
        X = np.array(self.df.drop(columns=['Hogwarts House']))
        X = np.hstack((np.ones((len(X), 1)), X))
        y = np.array(self.df['Hogwarts House'])
        y = y.reshape(y.shape[0], 1)
        learning_rate = 0.01
        iterations = 3000
        reg = LogReg(X, y, learning_rate, iterations, houses)
        params, self.cost_history = reg.gradient_descent()
        self.save_values(params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-c", "--cost_history", help="the csv file containing the data set", action="store_true")
    args = parser.parse_args()
    train = Train(args.data_file)
    train.read_data()
    train.train()
    if args.cost_history:
        plot_cost_history(train.cost_history)

if __name__ == "__main__":
    main()