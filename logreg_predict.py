#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import json
import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt

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

class Train:
    def __init__(self, data_file, weights_file):
        self.data_file = data_file
        self.weigths = self.read_weights(weights_file)
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

    def compute(self, row):
        row = row
        return ('Gryffindor')

    def predict(self):
        df = pd.read_csv(self.data_file)
        indexes = []
        houses = []
        for _, row in df.iterrows():
            indexes.append(row['Index'])
            houses.append(self.compute(row))
        return indexes, houses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_csv_file(parser, x))
    parser.add_argument("weights_file", help="the json file containing the weigths to use for the prediction", type=lambda x: is_valid_json_file(parser, x))
    args = parser.parse_args()
    train = Train(args.data_file, args.weights_file)
    train.save_values()

if __name__ == "__main__":
    main()