#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/10 11:19
"""

import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

from awesomeml.pth import FILE_PATH


style.use('ggplot')


def run():
    df = quandl.get("WIKI/GOOGL")
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100.0

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume', ]]
    # print(df.head())

    forecast_col = 'Adj. Close'
    df.fillna(-99999, inplace=True)

    forecast_out = int(math.ceil(0.1 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], axis=1))

    X = preprocessing.scale(X)
    X = X[:-forecast_out]
    X_lately = X[-forecast_out:]
    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = LinearRegression()
    # clf = LinearRegression(n_jobs=5)
    # clf = svm.SVR()
    # clf = svm.SVR(kernel='poly')
    clf.fit(X_train, y_train)

    # save model
    with open('{}/line_reg.pickle'.format(FILE_PATH),'wb') as fw:
        pickle.dump(clf,fw)

    with open('{}/line_reg.pickle'.format(FILE_PATH), 'rb') as f:
        clf = pickle.load(f)

    accuracy = clf.score(X_test, y_test)
    forecast_set = clf.predict(X_lately)

    print(forecast_set, accuracy, forecast_out)

    df['Forecast'] = np.nan
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.xlabel('Price')
    plt.show()


def tmp():
    data = [[1, 2, 3], [4, 5, 6]]
    index = ['idx1', 'idx2']
    columns = ['a', 'b', 'c']
    df = pd.DataFrame(data=data, index=index, columns=columns)

    print(df.loc['idx1'])
    print(df.iloc[1])


def main():
    run()
    # tmp()


if __name__ == '__main__':
    main()
