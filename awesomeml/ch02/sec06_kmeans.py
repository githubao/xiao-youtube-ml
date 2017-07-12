#!/usr/bin/env python
# encoding: utf-8

"""
@description: kmeans

@author: BaoQiang
@time: 2017/7/12 20:22
"""

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import cross_validation, preprocessing
from awesomeml.pth import FILE_PATH

input_file = '{}/titanic.xls'.format(FILE_PATH)


def run2():
    df = pd.read_excel(input_file)
    # print(df.head())
    df.drop(['body', 'name'], 1, inplace=True)

    df.drop(['ticket'], 1, inplace=True)

    df.convert_objects(convert_numeric=True)
    # for column in df.columns.values:
    #     df[column] = list(map(pd.to_numeric, df[column]))
    df.fillna(0, inplace=True)

    df = handle_non_numerical_data(df)
    # print(df.head())

    X = np.array(df.drop(['survived'], 1)).astype(float)
    # 归一化
    X = preprocessing.scale(X)

    y = np.array(df['survived'])

    clf = KMeans(n_clusters=2)
    clf.fit(X)

    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i]).astype(float)
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = clf.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    print(correct / len(X))


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = 0
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def run():
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], ])
    # plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5)
    # plt.show()

    clf = KMeans(n_clusters=2)
    clf.fit(X)

    centers = clf.cluster_centers_
    labels = clf.labels_

    colors = ['g.', 'r.', 'c.', 'b.', 'k.', 'o.']
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=150, linewidths=5)
    plt.show()


def main():
    # run()
    run2()


if __name__ == '__main__':
    main()
