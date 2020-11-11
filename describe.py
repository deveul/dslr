#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import math
from tabulate import tabulate

# from decimal import Decimal

def is_float(string):
    try:
        if string == '':
            return True
        if string.lower() == 'nan':
            return False
        float(string)
        return True
    except:
        return False

def to_float(string):
    if string == '':
        return float('nan')
    try:
        return float(string)
    except:
        print("Sorry, I expected a float and got {}".format(string))
        exit()

def dslr_sum(values):
    my_sum = float(0)
    count = len(values)
    for item in values:
        if not np.isnan(item):
            my_sum += item
        else:
            count -= 1
    return [count, my_sum / count]

def calculate_std(values, mean, count):
    my_sum = 0
    for value in values:
        if not np.isnan(value):
            my_sum += (value - mean) ** 2
    return (my_sum / (count - 1)) ** 0.5

def get_quantile(column, count):
    c = sorted([float(y) for y in column if float(y) == float(y)])
    minimum = c[0]
    maximum = c[count - 1]
    quarter = c[math.ceil(0.25 * count) - 1]
    median = c[math.ceil(0.5 * count) - 1]
    three_quarter = c[math.ceil(0.75 * count) - 1]
    return minimum, maximum, quarter, median, three_quarter

def analyse_data(data_file):
    df = pd.read_csv(data_file)
    describe = np.array(['count', 'mean', 'std', 'minimum', '25%', '50%', '75%', 'maximum']).reshape(8, 1)
    np.set_printoptions(suppress=True)
    headers = ['']
    for column in df.columns:
        if is_numeric_dtype(df[column].dtypes):
            headers.append(column)
            count, mean = dslr_sum(df[column])
            std = calculate_std(df[column], mean, count)
            minimum, maximum, quarter, median, three_quarter = get_quantile(df[column], count)
            # complete = [count, mean, std, minimum, quarter, median, three_quarter, maximum]
            # for index, _ in enumerate(complete):
                # complete[index] = [format(complete[index], '.6f')]
                # complete[index] = [complete[index]]
            new_column = [[count], [mean], [std], [minimum], [quarter], [median], [three_quarter], [maximum]]
            describe = np.append(describe, new_column, axis=1)
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
    args = parser.parse_args()
    describe, headers = analyse_data(args.data_file)
    # pd.set_option('display.expand_frame_repr', False)
    # print(pd.DataFrame(data=describe, columns=headers).to_string(index=False))
    print(tabulate(describe, headers, tablefmt="plain", floatfmt=".6f"))

if __name__ == "__main__":
    main()