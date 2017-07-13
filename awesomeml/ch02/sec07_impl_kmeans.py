#!/usr/bin/env python
# encoding: utf-8

"""
@description: 自己实现kmeans算法

@author: BaoQiang
@time: 2017/7/13 12:30
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=1000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centers = {}

        for i in range(self.k):
            self.centers[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centers[center]) for center in self.centers]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centers = dict(self.centers)

            for classification in self.classifications:
                self.centers[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centers:
                original_center = prev_centers[c]
                current_center = self.centers[c]
                if np.sum((current_center - original_center) / original_center * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centers[center]) for center in self.centers]
        classification = distances.index(min(distances))
        return classification


def run():
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], ])

    clf = KMeans()
    clf.fit(X)

    colors = 10 * ['g', 'r', 'c', 'b', 'k', 'o']
    for center in clf.centers:
        plt.scatter(clf.centers[center][0], clf.centers[center][1], marker='o', color='k', s=150, linewidths=5)

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

    unknowns = np.array([[1, 3], [8, 9], [0, 3], [5, 4], [6, 4], ])
    for unknown in unknowns:
        classification = clf.predict(unknown)
        plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification], s=150, linewidths=5)

    plt.show()


def main():
    run()


if __name__ == '__main__':
    main()
