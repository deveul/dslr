#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import json
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt
from utils.stats_functions import dslr_mean
from utils.stats_functions import dslr_std

def is_valid_csv_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def is_valid_json_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.json'):
        parser.error("The file {} has no json extension!".format(arg))
    else:
        return arg

class Predict:
    def __init__(self, data_file, weights_file):
        self.data_file = data_file
        self.gryffindor = None
        self.slytherin = None
        self.ravenclaw = None
        self.hufflepuff = None
        self.weigths = self.read_weights(weights_file)
        self.index = 0
        self.indexes, self.houses = self.predict()

    def read_weights(self, weights_file):
        weights = None
        try:
            with open(weights_file, 'r') as json_file:
                weights = json.load(json_file)
        except FileNotFoundError:
            print("Le fichier {} ne semble pas exister.".format(weights_file))
            exit()
        except PermissionError:
            print("Vous n'avez pas les droits pour lire le fichier value_lr.json")
            exit()
        except:
            print("Erreur inconnue")
            exit()
        self.gryffindor = np.array(weights["Gryffindor"])
        self.slytherin = np.array(weights["Slytherin"])
        self.ravenclaw = np.array(weights["Ravenclaw"])
        self.hufflepuff = np.array(weights["Hufflepuff"])
        return weights

    def save_values(self):
        df = pd.DataFrame({'Index': self.indexes,
                    'Hogwarts House': self.houses})
        try:
            df.to_csv('./houses.csv', index=False)
            print("houses.csv created or updated")
        except PermissionError:
            print("Vous n'avez pas les droits pour écrire dans le fichier houses.json")
            exit()
        except:
            print("Erreur inconnue")
            exit()

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compute(self, row):
        gryf = np.dot(row, self.gryffindor)
        slyt = np.dot(row ,self.slytherin)
        rave = np.dot(row, self.ravenclaw)
        huff = np.dot(row, self.hufflepuff)
        if gryf > max([slyt, rave, huff]):
            return "Gryffindor"
        elif slyt > max([gryf, rave, huff]):
            return "Slytherin"
        elif rave > max([gryf, slyt, huff]):
            return "Ravenclaw"
        else:
            return "Hufflepuff"

    def standardize_values(self, X):
        X = (X - dslr_mean(X)) / dslr_std(X)
        return X

    def predict(self):
        df = pd.read_csv(self.data_file, usecols=['Astronomy', 'Herbology', 'Ancient Runes'])
        for column in df:
            if is_numeric_dtype(df[column].dtypes):
                df[column] = df[column].fillna(dslr_mean(df[column].dropna()))
                df[column] = self.standardize_values(df[column])
        X = np.array(df)
        X = np.hstack((np.ones((len(X), 1)), X))
        indexes = []
        houses = []
        for i, row in enumerate(X):
            indexes.append(i)
            houses.append(self.compute(row))
        return indexes, houses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_csv_file(parser, x))
    parser.add_argument("weights_file", help="the json file containing the weigths to use for the prediction", type=lambda x: is_valid_json_file(parser, x))
    args = parser.parse_args()
    predict = Predict(args.data_file, args.weights_file)
    predict.save_values()

if __name__ == "__main__":
    main()