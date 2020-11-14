#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
# from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def read_data(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(columns=['Index', 'Arithmancy', 'Care of Magical Creatures', 'Defense Against the Dark Arts'])
    palette={"Gryffindor": "r", "Slytherin": "darkgreen", "Ravenclaw": "royalblue", "Hufflepuff": "gold"}
    sns.pairplot(df, hue="Hogwarts House", height=2, plot_kws={"s": 3}, palette=palette)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()
    read_data(args.data_file)

if __name__ == "__main__":
    main()