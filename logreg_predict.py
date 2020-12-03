#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import json
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import pyplot as plt
from utils.log_reg import predict
from utils.stats_functions import dslr_mean
from utils.stats_functions import dslr_std
from utils.stats_functions import dslr_max
from utils.stats_functions import z_score

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
        self.df = None
        self.gryffindor = None
        self.slytherin = None
        self.ravenclaw = None
        self.hufflepuff = None
        self.weigths = self.read_weights(weights_file)
        self.indexes, self.houses = self.predict_all()

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
        try:
            self.gryffindor = np.array(weights["Gryffindor"]).reshape(4, 1)
            self.slytherin = np.array(weights["Slytherin"]).reshape(4, 1)
            self.ravenclaw = np.array(weights["Ravenclaw"]).reshape(4, 1)
            self.hufflepuff = np.array(weights["Hufflepuff"]).reshape(4, 1)
        except:
            print("Wrong formatage of the wieght file")
            exit()
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

    def predict_all(self):
        df = pd.read_csv(self.data_file, usecols=['Astronomy', 'Herbology', 'Ancient Runes'])
        for column in df:
            if is_numeric_dtype(df[column].dtypes):
                df[column] = df[column].fillna(dslr_mean(df[column].dropna()))
                df[column] = z_score(df[column])
        self.df = df
        X = np.array(df)
        X = np.hstack((np.ones((len(X), 1)), X))
        gryf = predict(X, self.gryffindor)
        slyt = predict(X, self.slytherin)
        rave = predict(X, self.ravenclaw)
        huff = predict(X, self.hufflepuff)
        data = np.hstack((gryf, slyt, rave, huff))
        df = pd.DataFrame(data, columns=["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"])
        houses = df.idxmax(axis=1)
        return range(len(houses)), houses

    def get_house_repartition(self, house, dfs):
        x = dfs['Astronomy'].get_group(house)
        y = dfs['Herbology'].get_group(house)
        z = dfs['Ancient Runes'].get_group(house)
        return x, y, z

    def visualize(self):
        houses = pd.DataFrame(self.houses, columns=['House'])
        repartition = pd.concat([houses, self.df], axis=1)
        dfs = repartition.groupby('House')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("Students repartition")

        xg, yg, zg = self.get_house_repartition("Gryffindor", dfs)
        xs, ys, zs = self.get_house_repartition("Slytherin", dfs)
        xr, yr, zr = self.get_house_repartition("Ravenclaw", dfs)
        xh, yh, zh = self.get_house_repartition("Hufflepuff", dfs)

        ax.scatter(xg, yg, zg, label='Gryffindor',c='r')
        ax.scatter(xs, ys, zs, label='Slytherin',c='darkgreen')
        ax.scatter(xr, yr, zr, label='Ravenclaw',c='b')
        ax.scatter(xh, yh, zh, label='Hufflepuff',c='gold')

        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.10, 0.5), fontsize=9)

        ax.set_xlabel('\nAstronomy')
        ax.set_ylabel('\nHerbology')
        ax.set_zlabel('\nAncient Runes')

        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_csv_file(parser, x))
    parser.add_argument("weights_file", help="the json file containing the weigths to use for the prediction", type=lambda x: is_valid_json_file(parser, x))
    parser.add_argument("-v", "--visualize", help="vizualize repartition of the students in 3d", action="store_true")
    args = parser.parse_args()
    predict = Predict(args.data_file, args.weights_file)
    predict.save_values()
    if args.visualize:
            predict.visualize()

if __name__ == "__main__":
    main()