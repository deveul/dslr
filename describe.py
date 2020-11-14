#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import math
from tabulate import tabulate
import scipy.stats as stats
import statistics

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

def calculate_std_var(values, mean, count):
    my_sum = 0
    for value in values:
        if not np.isnan(value):
            my_sum += (value - mean) ** 2
    var = my_sum / (count - 1)
    std = var ** 0.5
    return std, var

def get_quantile(column, count):
    c = sorted([float(y) for y in column if float(y) == float(y)])
    minimum = c[0]
    maximum = c[count - 1]
    index_quarter = (count - 1) / 4
    quarter = None
    if index_quarter.is_integer():
        quarter = c[int(index_quarter)]
    else:
        if round(index_quarter) - index_quarter == 0.5 or round(index_quarter) - index_quarter == -0.5:
            upper = c[math.ceil(index_quarter)]
            lower = c[math.floor(index_quarter)]
            quarter = (lower + upper) / 2
        elif round(index_quarter) - index_quarter > 0:
            upper = c[math.ceil(index_quarter)]
            lower = c[math.floor(index_quarter)]
            quarter = (lower + 3 * upper) / 4
        else:
            upper = c[math.ceil(index_quarter)]
            lower = c[math.floor(index_quarter)]
            quarter = (3 * lower + upper) / 4
    index_median = (count - 1) / 2
    median = None
    if index_median.is_integer():
        median = c[int(index_median)]
    else:
        upper = c[math.ceil(index_median)]
        lower = c[math.floor(index_median)]
        median = (lower + upper) / 2
    index_three_quarter = (count - 1) * 0.75
    three_quarter = None
    if index_three_quarter.is_integer():
        three_quarter = c[int(index_three_quarter)]
    else:
        if round(index_three_quarter) - index_three_quarter == 0.5 or round(index_three_quarter) - index_three_quarter == -0.5:
            upper = c[math.ceil(index_three_quarter)]
            lower = c[math.floor(index_three_quarter)]
            three_quarter = (lower + upper) / 2
        elif round(index_three_quarter) - index_three_quarter > 0:
            upper = c[math.ceil(index_three_quarter)]
            lower = c[math.floor(index_three_quarter)]
            three_quarter = (lower + 3 * upper) / 4
        else:
            upper = c[math.ceil(index_three_quarter)]
            lower = c[math.floor(index_three_quarter)]
            three_quarter = (3 * lower + upper) / 4
    return minimum, maximum, quarter, median, three_quarter

def calculate_skewness_kurtosis(values, mean, std, count):
    sum_skew = 0
    sum_kurt = 0
    sum_kurt_deno = 0
    for value in values:
        if not np.isnan(value):
            sum_skew += (value - mean) ** 3
            sum_kurt += (value - mean) ** 4
            sum_kurt_deno += (value - mean) ** 2
    skewness = sum_skew / ((count - 1) * (std ** 3))
    kurtosis_num = sum_kurt / count
    kurtosis_deno = (sum_kurt_deno / count) ** 2
    kurtosis = kurtosis_num / kurtosis_deno - 3
    return skewness, kurtosis

def calculate_median_absolute_deviation(values, median, count):
    new_values = sorted([abs(x - median) for x in values])
    index_median = (count - 1) / 2
    median_absolute_deviation = None
    if index_median.is_integer():
        median_absolute_deviation = new_values[int(index_median)]
    else:
        lower = new_values[math.floor(index_median)]
        upper = new_values[math.ceil(index_median)]
        median_absolute_deviation = (lower + upper) / 2
    return median_absolute_deviation

def calculate_mean_absolute_deviation(values, mean, count):
    sum_mean_ad = 0
    for value in values:
        if not np.isnan(value):
            distance = value - mean
            if distance < 0:
                distance = distance * - 1
            sum_mean_ad += distance
    mean_absolute_deviation = sum_mean_ad / count
    return mean_absolute_deviation

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
            count, mean = dslr_sum(df[column])
            std, var = calculate_std_var(df[column], mean, count)
            minimum, maximum, quarter, median, three_quarter = get_quantile(df[column].dropna(), count)
            new_column = [[count], [mean], [std], [minimum], [quarter], [median], [three_quarter], [maximum]]
            describe = np.append(describe, new_column, axis=1)
            if advanced:
                skew, kurt = calculate_skewness_kurtosis(df[column], mean, std, count)
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
    # pd.set_option('display.expand_frame_repr', False)
    # print(pd.DataFrame(data=describe, columns=headers).to_string(index=False))
    print(tabulate(describe, headers, tablefmt="github", floatfmt=".6f"))

if __name__ == "__main__":
    main()