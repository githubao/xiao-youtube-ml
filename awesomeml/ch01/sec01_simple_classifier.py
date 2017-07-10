#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/6 18:42
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import requests
import pydot
from scipy.spatial import distance
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score

from awesomeml.pth import FILE_PATH

from tensorflow.contrib import learn

tf.logging.set_verbosity(tf.logging.ERROR)


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


def run5():
    # IRIS_TRAINING = "iris_training.csv"
    # IRIS_TEST = "iris_test.csv"

    iris = learn.datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]
    clf = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feature_columns)

    clf.fit(X_train, y_train, steps=200)
    score = accuracy_score(y_test, clf.predict(X_test))
    print('Accuracy: {0:f}'.format(score))


def run6():
    '''
    Test Accuracy: 0.966667
    New Samples, Class Predictions:    [1, 1]
    '''

    IRIS_TRAINING = "{}/iris_training.csv".format(FILE_PATH)
    IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
    IRIS_TEST = "{}/iris_test.csv".format(FILE_PATH)
    IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = requests.get(IRIS_TRAINING_URL).content
        with open(IRIS_TRAINING, "wb") as f:
            f.write(raw)
    if not os.path.exists(IRIS_TEST):
        raw = requests.get(IRIS_TEST_URL).content
        with open(IRIS_TEST, "wb") as f:
            f.write(raw)
    # Load datasets.
    training_set = learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = learn.DNNClassifier(feature_columns=feature_columns,
                                     hidden_units=[10, 20, 10],
                                     n_classes=3,
                                     model_dir="{}/iris_model".format(FILE_PATH))

    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

        # Fit model.

    classifier.fit(input_fn=get_train_inputs, steps=2000)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)
        return x, y

        # Evaluate accuracy.

    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    def new_samples():
        return np.array(
            [[6.4, 3.2, 4.5, 1.5],
             [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print(
        "New Samples, Class Predictions:    {}\n"
            .format(predictions))


def run7():
    mnist = learn.datasets.load_dataset('mnist')
    data = mnist.train.images
    labels = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    max_examples = 1000
    data = data[:max_examples]
    labels = labels[:max_examples]

    def display(i):
        img = test_data[i]
        plt.title('Example {}, Label: {}'.format(i, test_labels[i]))
        plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray)
        plt.show()

    # display(0)

    feature_columns = learn.infer_real_valued_columns_from_input(data)
    clf = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10)
    clf.fit(data, labels, batch_size=100, steps=1000)

    "0.8607"
    print(clf.evaluate(test_data, test_labels)['accuracy'])

    # print('Predicted {}, Label: {}'.format(clf.predict(test_data[0]), test_labels[0]))

    weights = clf.weights_

    f, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.reshape(-1)
    for i in range(len(axes)):
        a = axes[i]
        a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
        a.set_title(i)
        a.set_xticks(())
        a.set_yticks(())

    plt.show()


def main():
    # run()
    # run2()
    # run3()
    run4()
    # run6()
    # run7()


if __name__ == '__main__':
    main()
