#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import json
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

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
        self.coef_0 = 0.3
        self.coef_1 = 4
        self.coef_2 = 1

    def read_data(self):
        pd.read_csv(self.data_file)

    def save_values(self):
        coefficients = {
            'coef_0': self.coef_0,
            'coef_1': self.coef_1,
            'coef_2': self.coef_2,
        }
        try:
            with open('value_logreg.json', 'w') as json_file:
                json.dump(coefficients, json_file)
                print("value_logreg.json updated")
        except PermissionError:
            print("Vous n'avez pas les droits pour écrire dans le fichier value_lr.json")
            exit()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def square_loss(self, y_pred, target):
        return np.mean(pow((y_pred - target),2))

    def train(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()
    train = Train(args.data_file)
    train.read_data()
    train.train()
    train.save_values()

if __name__ == "__main__":
    main()