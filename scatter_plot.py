#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def read_data(data_file):
    df = pd.read_csv(data_file)
    plt.scatter([x for x in df['Astronomy']], [y for y in df['Defense Against the Dark Arts']], c='blue', alpha=0.5, label='Students')
    plt.title('Correlated features')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()
    read_data(args.data_file)

if __name__ == "__main__":
    main()