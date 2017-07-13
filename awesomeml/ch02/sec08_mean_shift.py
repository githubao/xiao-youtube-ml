#!/usr/bin/env python
# encoding: utf-8

"""
@description: mean shift 算法

@author: BaoQiang
@time: 2017/7/13 13:10
"""

from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class MyMeanShift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centers = {}

        for i in range(len(data)):
            centers[i] = data[i]

        while True:
            new_centers = []
            for i in centers:
                in_bandwidth = []
                center = centers[i]
                for featureset in data:
                    if np.linalg.norm(featureset - center) < self.radius:
                        in_bandwidth.append(featureset)

                new_center = np.average(in_bandwidth, axis=0)
                new_centers.append(tuple(new_center))

            uniques = sorted(list(set(new_centers)))

            prev_centers = dict(centers)

            centers = {}
            for i in range(len(uniques)):
                centers[i] = np.array(uniques[i])

            optimized = True
            for i in centers:
                if not np.array_equal(centers[i], prev_centers[i]):
                    optimized = False

                if not optimized:
                    break

            if optimized:
                break

        self.centers = centers

    def predict(self, data):
        pass


def run2():
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],[8, 2], [10, 2], [9, 3], ])
    colors = 10 * ['g', 'r', 'c', 'b', 'k']

    clf = MyMeanShift()
    clf.fit(X)
    centers = clf.centers
    plt.scatter(X[:, 0], X[:, 1], s=150)

    for c in centers:
        plt.scatter(centers[c][0], centers[c][1], color='k', marker='*', s=150)

    plt.show()


def run():
    centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10], ]
    X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=1)
    ms = MeanShift()
    ms.fit(X)
    lables = ms.labels_
    cluster_centers = ms.cluster_centers_

    print(cluster_centers)
    n_clusters = len(np.unique(lables))
    print('Number of estimated clusters: ', n_clusters)

    colors = ['g', 'r', 'c', 'b', 'k', 'o']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[lables[i]], marker='o')

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], color='k', marker='x', s=150,
               linewidths=5, zorder=10)

    plt.show()


def main():
    # run()
    run2()


if __name__ == '__main__':
    main()
