#!/usr/bin/python3
# -*-coding:Utf-8 -*

import numpy as np
import math

def covariance(x, y, x_mean, y_mean):
    return np.sum((x - x_mean) * (y - y_mean)) / (len(x) - 1)

def pearson_core(x, y):
    _, y_mean = dslr_sum(y.dropna())
    _, x_mean = dslr_sum(x.dropna())
    x = x.fillna(x_mean)
    y = y.fillna(y_mean)
    cov = covariance(x, y, x_mean, y_mean)
    x_std, _ = calculate_std_var(x, x_mean, len(x))
    y_std, _ = calculate_std_var(y, y_mean, len(y))
    return cov / (x_std * y_std)

def dslr_sum(values):
    my_sum = float(0)
    count = len(values)
    for item in values:
        my_sum += item
    return [count, my_sum / count]

def calculate_std_var(values, mean, count):
    my_sum = 0
    for value in values:
        my_sum += (value - mean) ** 2
    var = my_sum / (count - 1)
    std = var ** 0.5
    return std, var

def calculate_skewness_kurtosis(values, mean, std, count):
    sum_skew = 0
    sum_kurt = 0
    sum_kurt_deno = 0
    for value in values:
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

def get_quantile(column, count):
    c = sorted(column)
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
