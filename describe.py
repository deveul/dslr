#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import math
from tabulate import tabulate
from utils.stats_functions import dslr_mean
from utils.stats_functions import dslr_min
from utils.stats_functions import dslr_max
from utils.stats_functions import dslr_std
from utils.stats_functions import dslr_var
from utils.stats_functions import dslr_quantile
from utils.stats_functions import dslr_skewness
from utils.stats_functions import dslr_kurtosis
from utils.stats_functions import dslr_median_absolute_deviation
from utils.stats_functions import dslr_mean_absolute_deviation

def analyse_data(data_file, advanced):
    df = pd.read_csv(data_file)
    describe = np.array(['count', 'mean', 'std', 'minimum', '25%', '50%', '75%', 'maximum']).reshape(8, 1)
    if advanced:
        describe_advanced = np.array(['range', 'var', 'skew', 'kurt', 'mad', 'mean_ad']).reshape(6, 1)
    np.set_printoptions(suppress=True)
    headers = ['']
    for column in df.columns:
        if is_numeric_dtype(df[column].dtypes):
            headers.append(column)
            count = len(df[column].dropna())
            mean = dslr_mean(df[column].dropna())
            std = dslr_std(df[column].dropna())
            minimum = dslr_min(df[column].dropna())
            maximum = dslr_max(df[column].dropna())
            quarter, median, three_quarter = dslr_quantile(df[column].dropna())
            new_column = [[count], [mean], [std], [minimum], [quarter], [median], [three_quarter], [maximum]]
            describe = np.append(describe, new_column, axis=1)
            if advanced:
                var = dslr_var(df[column].dropna())
                skew = dslr_skewness(df[column].dropna())
                kurt = dslr_kurtosis(df[column].dropna())
                mad = dslr_median_absolute_deviation(df[column].dropna())
                mean_ad = dslr_mean_absolute_deviation(df[column].dropna())
                describe_advanced = np.append(describe_advanced, [[maximum - minimum], [var], [skew], [kurt], [mad], [mean_ad]], axis=1)
    if advanced:
        describe = np.append(describe, describe_advanced, axis=0)
    return describe, headers

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def save_output(filename, output):
    filename = filename + "-dslr.txt"
    try:
        with open(filename, 'w') as output_file:
            output_file.write(output)
    except:
        print("Error trying to write to the file {}".format(filename))
        exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-a", "--advanced", help="calculate other useful statistics on each columm", action="store_true")
    parser.add_argument("-o", "--out", help="save the output to the specified file", nargs='?', const="describe", action="store")
    args = parser.parse_args()
    describe, headers = analyse_data(args.data_file, args.advanced)
    output = tabulate(describe, headers, tablefmt="github", floatfmt=".6f")
    if args.out:
        save_output(args.out, output)
    else:
        print(output)

if __name__ == "__main__":
    main()