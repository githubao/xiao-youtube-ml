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
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):
        if self.radius is None:
            all_data_center = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_center)

            self.radius = all_data_norm / self.radius_norm_step

        centers = {}

        for i in range(len(data)):
            centers[i] = data[i]

        while True:
            new_centers = []
            for i in centers:
                in_bandwidth = []
                center = centers[i]

                weights = [i for i in range(self.radius_norm_step)][::-1]

                for featureset in data:
                    distance = np.linalg.norm(featureset - center)
                    if distance == 0:
                        distance = 0.00000001
                    weight_index = int(distance / self.radius)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1

                    to_add = (weights[weight_index] ** 2) * [featureset]
                    in_bandwidth += to_add


                    # if np.linalg.norm(featureset - center) < self.radius:
                    #     in_bandwidth.append(featureset)

                new_center = np.average(in_bandwidth, axis=0)
                new_centers.append(tuple(new_center))

            uniques = sorted(list(set(new_centers)))

            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

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
        self.classifications = {}
        for i in range(len(self.centers)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centers[center]) for center in self.centers]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centers[center]) for center in self.centers]
        classification = distances.index(min(distances))
        return classification


def run2():
    # X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3], ])

    X, y = make_blobs(n_samples=20, centers=3, n_features=2)

    colors = 10 * ['g', 'r', 'c', 'b', 'k']

    clf = MyMeanShift()
    clf.fit(X)
    centers = clf.centers
    # plt.scatter(X[:, 0], X[:, 1], s=150)

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

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
