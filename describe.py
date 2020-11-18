#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import math
from tabulate import tabulate
from utils.stats_functions import dslr_sum
from utils.stats_functions import calculate_std_var
from utils.stats_functions import get_quantile
from utils.stats_functions import calculate_skewness_kurtosis
from utils.stats_functions import calculate_median_absolute_deviation
from utils.stats_functions import calculate_mean_absolute_deviation

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
            count, mean = dslr_sum(df[column].dropna())
            std, var = calculate_std_var(df[column].dropna(), mean, count)
            minimum, maximum, quarter, median, three_quarter = get_quantile(df[column].dropna(), count)
            new_column = [[count], [mean], [std], [minimum], [quarter], [median], [three_quarter], [maximum]]
            describe = np.append(describe, new_column, axis=1)
            if advanced:
                skew, kurt = calculate_skewness_kurtosis(df[column].dropna(), mean, std, count)
                mad = calculate_median_absolute_deviation(df[column].dropna(), median, count)
                mean_ad = calculate_mean_absolute_deviation(df[column].dropna(), mean, count)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-a", "--advanced", help="calculate other useful statistics on each columm", action="store_true")
    args = parser.parse_args()
    describe, headers = analyse_data(args.data_file, args.advanced)
    print(tabulate(describe, headers, tablefmt="github", floatfmt=".6f"))

if __name__ == "__main__":
    main()