#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/6 18:42
"""

from sklearn import tree
import numpy as np

from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydot
from awesomeml.pth import FILE_PATH

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from scipy.spatial import distance

import random

from sklearn.metrics import accuracy_score


def run():
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [0, 0, 1, 1]
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)
    # test = [140, 1]
    test = np.array([140, 0])
    print(clf.predict(test.reshape((1, -1))))


def run2():
    iris = load_iris()

    test_idx = [0, 50, 100]
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)

    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    print(iris.feature_names)
    print(iris.target_names)

    print(test_data)
    print(test_target)

    print(clf.predict(test_data))
    print(test_target)

    dot_data = StringIO()

    tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names,
                         filled=True, rounded=True, impurity=False)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf('{}/iris_decision-tree.pdf'.format(FILE_PATH))


def run3():
    greyhounds = 500
    labs = 500

    grey_height = 28 + 4 * np.random.randn(greyhounds)
    lab_height = 24 + 4 * np.random.randn(labs)

    plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
    plt.show()


def euc(a, b):
    return distance.euclidean(a, b)


class ScrapyKnn():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            # label = random.choice(self.y_train)
            label = self.closest(row)

            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


def run4():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # clf = KNeighborsClassifier()
    # clf = tree.DecisionTreeClassifier()
    clf = ScrapyKnn()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)


def main():
    # run()
    # run2()
    # run3()
    run4()


if __name__ == '__main__':
    main()
