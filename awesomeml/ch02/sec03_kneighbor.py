#!/usr/bin/env python
# encoding: utf-8

"""
@description: K 临近算法

@author: BaoQiang
@time: 2017/7/10 17:18
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors,svm
import pandas as pd
from awesomeml.pth import FILE_PATH
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import style
import warnings
import random

# bmh ggplot dark_background fivethirtyeight,grayscale
style.use('fivethirtyeight')

input_file = '{}/breast-cancer-wisconsin.data.txt'.format(FILE_PATH)


def run():
    df = pd.read_csv(input_file)
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    # print(df.head())

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    # clf = neighbors.KNeighborsClassifier()
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    """
    0.964285714286
    """
    print(accuracy)

    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 1, 2, 3, 2, 1]])
    example_measures = example_measures.reshape((len(example_measures), -1))
    print(clf.predict(example_measures))


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]

    counter = Counter(votes).most_common(1)
    vote_result = counter[0][0]
    confidence = counter[0][1] / k
    # print('Confidence: ',confidence)

    return vote_result, confidence


def run2():
    dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    new_features = [5, 7]

    for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color=i)
    plt.scatter(new_features[0], new_features[1])
    plt.show()

    result = k_nearest_neighbors(dataset, new_features, k=3)
    print(result)


def run3():
    df = pd.read_csv(input_file)
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    full_data = df.astype(float).values.tolist()
    # print(full_data[:5])

    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}

    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            else:
                print(confidence)
            total += 1

    # Accuracy:  0.9280575539568345
    print('Accuracy: ', correct / total)

    return correct / total


def run4():
    num_works = 3
    accuracies = []
    for i in range(num_works):
        accuracies.append(run3())

    print('avg acc: ', sum(accuracies) / len(accuracies))


def main():
    run()
    # run2()
    # run3()
    # run4()


if __name__ == '__main__':
    main()
