#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/10 13:04
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(hm, variance, step=2, correlation=''):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


xs, ys = create_dataset(40, 10, 2, correlation='pos')


def get_best_slope_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys))) / (np.power(mean(xs), 2) - mean(xs * xs))
    b = mean(ys) - m * mean(xs)
    return m, b


def plot():
    plt.scatter(xs, ys)
    plt.show()


def plot2(x, y):
    plt.scatter(xs, ys)
    plt.plot(x, y)
    plt.show()


def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line) ** 2)


# 越大越好
def coefficient_of_determination(ys_orig, ys_line):
    """
    1 - (原始值-预测值)的平方和 / (原始值-平均值)的平方和
    :param ys_orig: 
    :param ys_line: 
    :return: 
    """

    y_mean_line = [mean(ys_orig) for _ in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


def run():
    m, b = get_best_slope_intercept(xs, ys)
    y_ = [m * x + b for x in xs]
    plot2(xs, y_)

    print(m, b)

    pred_x = 8
    pred_y = m * pred_x + b

    print(pred_y)

    r_squared = coefficient_of_determination(ys, y_)
    print(r_squared)


def main():
    run()
    # plot()


if __name__ == '__main__':
    main()
