#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt
import seaborn as sns

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def read_data(data_file, content):
    try:
        df = None
        if content == "train":
            df = pd.read_csv(data_file, usecols=['Hogwarts House', 'Astronomy', 'Herbology', 'Ancient Runes'])
        elif content == "normal":
            df = pd.read_csv(data_file, usecols=lambda col: col not in ['Index', 'Arithmancy', 'Care of Magical Creatures', 'Defense Against the Dark Arts'])
        else:
            df = pd.read_csv(data_file, usecols=lambda col: col not in ['Index'])
        palette={"Gryffindor": "r", "Slytherin": "darkgreen", "Ravenclaw": "royalblue", "Hufflepuff": "gold"}
        sns.pairplot(df, hue="Hogwarts House", height=1, plot_kws={"s": 3}, palette=palette)
        plt.show()
    except:
        print("Une erreur est survenue pendant l'exploitation du data set")
        exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-c", "--content", help="the pair_plot you want to display", choices=["train", "normal", "all"], default="normal")
    args = parser.parse_args()
    read_data(args.data_file, args.content)
    
if __name__ == "__main__":
    main()