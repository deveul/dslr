#!/usr/bin/python3
# -*-coding:Utf-8 -*

import numpy as np
import math

def z_score(values):
    X = np.array(values)
    return (X - dslr_mean(X)) / dslr_std(X)

def covariance(x, y, x_mean, y_mean):
    return np.sum((x - x_mean) * (y - y_mean)) / (len(x) - 1)

def pearson_core(x, y):
    y_mean = dslr_mean(y.dropna())
    x_mean = dslr_mean(x.dropna())
    x = x.fillna(x_mean)
    y = y.fillna(y_mean)
    cov = covariance(x, y, x_mean, y_mean)
    x_std = dslr_std(x)
    y_std = dslr_std(y)
    return cov / (x_std * y_std)

def dslr_mean(values):
    my_sum = float(0)
    for item in values:
        my_sum += item
    return my_sum / len(values)

def dslr_std(values):
    std = dslr_var(values) ** 0.5
    return std

def dslr_var(values):
    my_sum = 0
    values_mean = dslr_mean(values)
    for value in values:
        my_sum += (value - values_mean) ** 2
    var = my_sum / (len(values) - 1)
    return var

def dslr_skewness(values):
    mean = dslr_mean(values)
    std = dslr_std(values)
    count = len(values)
    sum_skew = 0
    for value in values:
        sum_skew += (value - mean) ** 3
    skewness = sum_skew / ((count - 1) * (std ** 3))
    return skewness

def dslr_kurtosis(values):
    mean = dslr_mean(values)
    count = len(values)
    sum_kurt = 0
    sum_kurt_deno = 0
    for value in values:
        sum_kurt += (value - mean) ** 4
        sum_kurt_deno += (value - mean) ** 2
    kurtosis_num = sum_kurt / count
    kurtosis_deno = (sum_kurt_deno / count) ** 2
    kurtosis = kurtosis_num / kurtosis_deno - 3
    return kurtosis

def dslr_median_absolute_deviation(values):
    median = dslr_quantile(values)[1]
    count = len(values)
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

def dslr_mean_absolute_deviation(values):
    mean = dslr_mean(values)
    count = len(values)
    sum_mean_ad = 0
    for value in values:
        if not np.isnan(value):
            distance = value - mean
            if distance < 0:
                distance = distance * - 1
            sum_mean_ad += distance
    mean_absolute_deviation = sum_mean_ad / count
    return mean_absolute_deviation

def dslr_min(values):
    values = sorted(values)
    return values[0]

def dslr_max(values):
    values = sorted(values)
    return values[len(values) - 1]

def dslr_quantile(column):
    c = sorted(column)
    count = len(c)
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
    return quarter, median, three_quarter
