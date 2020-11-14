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

def positive_int_type(arg):
    """ Type function for argparse - a int that must be positive or null """
    try:
        nb = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be an integer")
    if nb < 0 or nb > 100:
        raise argparse.ArgumentTypeError("Argument must be an int, from 1 to 100")
    return nb

def display_histogram(column, overlapping, bins, density, df, gryf, slyth, ravenclaw, hufflepuff):
    axes = plt.gca()
    y1=gryf[column]
    y2=slyth[column]
    y3=ravenclaw[column]
    y4=hufflepuff[column]
    colors = ['r', 'g', 'b', 'y']
    bins = bins
    if overlapping and density:
        plt.hist(y3, bins, color='b', alpha=0.5, density=True, label='Ravenclaw')
        plt.hist(y4, bins, color='y', alpha=0.5, density=True, label='Hufflepuff')
        plt.hist(y1, bins, color='r', alpha=0.5, density=True, label='Gryffindor')
        plt.hist(y2, bins, color='g', alpha=0.5, density=True, label='Slytherin')
    elif overlapping:
        plt.hist(y3, bins, color='b', alpha=0.5, label='Ravenclaw')
        plt.hist(y4, bins, color='y', alpha=0.5, label='Hufflepuff')
        plt.hist(y1, bins, color='r', alpha=0.5, label='Gryffindor')
        plt.hist(y2, bins, color='g', alpha=0.5, label='Slytherin')
    elif density:
        plt.hist([y1,y2, y3, y4], bins, density = True, color=colors, label=['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'])
    else:
        plt.hist([y1,y2, y3, y4], bins, color=colors, label=['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'])
    axes.set_xlim(min(df[column]), max(df[column]))
    axes.set_xlabel("Grades")
    axes.set_ylabel("Count by house")
    plt.title('Grades repartition for {}'.format(column))
    plt.legend()
    plt.show()

def read_data(data_file, all_classes, overlapping, bins, density):
    df = pd.read_csv(data_file)
    df_gryffindor = df[df['Hogwarts House'] == "Gryffindor"]
    df_slytherin = df[df['Hogwarts House'] == "Slytherin"]
    df_ravenclaw = df[df['Hogwarts House'] == "Ravenclaw"]
    df_hufflepuff = df[df['Hogwarts House'] == "Hufflepuff"]
    if all_classes == True:
        for index, column in enumerate(df.columns):
            if is_numeric_dtype(df[column].dtypes) and index > 0:
                display_histogram(column, overlapping, bins, density, df, df_gryffindor, df_slytherin, df_ravenclaw, df_hufflepuff)
    else:
        display_histogram('Arithmancy', overlapping, bins, density, df, df_gryffindor, df_slytherin, df_ravenclaw, df_hufflepuff)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-a", "--all_classes", help="display an histogram for each course", action="store_true")
    parser.add_argument("-o", "--overlapping", help="display the houses on top of each other", action="store_true")
    parser.add_argument("-d", "--density", help="y axis in percentage instead of number of students", action="store_true")
    parser.add_argument("-b", "--bins", help="Number of bins (intervals) per house", type=positive_int_type, default=10)
    args = parser.parse_args()
    read_data(args.data_file, args.all_classes, args.overlapping, args.bins, args.density)

if __name__ == "__main__":
    main()