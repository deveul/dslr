#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import json
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt
import scipy.stats as ss

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
        print(self.gryffindor)
        return weights

    def save_values(self):
        df = pd.DataFrame({'Index': self.indexes,
                    'Hogwarts House': self.houses})
        try:
            df.to_csv('./houses.csv', index=False)
            print("houses.csv updated")
        except PermissionError:
            print("Vous n'avez pas les droits pour écrire dans le fichier houses.json")
            exit()
        except:
            print("Erreur inconnue")
            exit()

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compute(self, row):
        # row = np.array(row)
        # row = np.hstack((row, 1))
        if self.index == 0:
            print(row)
            self.index += 1
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

    def predict(self):
        df = pd.read_csv(self.data_file, usecols=['Astronomy', 'Herbology', 'Ancient Runes'])
        df.fillna(df.mean(),inplace=True)
        X = np.array(df)
        X = np.array(ss.zscore(X))
        X = np.hstack((np.ones((len(X), 1)), X))
        # gryf = np.dot(X, self.gryffindor)
        # slyt = np.dot(X ,self.slytherin)
        # rave = np.dot(X, self.ravenclaw)
        # huff = np.dot(X, self.hufflepuff)
        # result = np.where(gryf >= slyt, "Gryffindor", "Slytherin")
        # result2 = np.where(rave >= gryf, "Ravenclaw", result)
        # result3 = np.where(huff >= , "Hufflepuff", result2)
        # result = np.max([gryf,slyt,rave,huff],axis=0, initial=-99999999, where=self.compare(gryf,slyt,rave,huff))
        # print(result[15:25])
        indexes = []
        houses = []
        i = 0
        for _, row in enumerate(X):
            indexes.append(i)
            i += 1
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