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


def main():
    # run()
    run2()


if __name__ == '__main__':
    main()
