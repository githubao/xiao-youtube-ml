#!/usr/bin/env python
# encoding: utf-8

"""
@description: 文本情感分析

@author: BaoQiang
@time: 2017/7/18 13:38
"""

"""
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
"""

import matplotlib.pyplot as plt
from matplotlib import style, animation
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

style.use('ggplot')

from nltk.classify import ClassifierI
from statistics import mode
import random

import nltk
import random
from nltk.corpus import movie_reviews
from awesomeml.pth import FILE_PATH


def gen():
    with open('{}/twitter-out.txt'.format(FILE_PATH), 'w') as fw:
        for i in range(100000):
            rand = random.randint(0, 2)
            if rand == 0:
                fw.write('neg\n')
            else:
                fw.write('pos\n')


def plot_twitter_trend():
    gen()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    def animate(i):
        with open('{}/twitter-out.txt'.format(FILE_PATH), 'r') as f:
            lines = f.readlines()
        xar = []
        yar = []

        x = 0
        y = 0

        for l in lines:
            x += 1
            if 'pos' in l:
                y += 1
            elif 'neg' in l:
                y -= 1

            xar.append(x)
            yar.append(y)

        ax1.clear()
        ax1.plot(xar, yar)

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


def new_training_data():
    documents = []
    with open('/short_reviews/positive.txt') as f:
        pos_datas = f.readlines()


class VoteClassifier(ClassifierI):
    def __init__(self, *classifers):
        self._classifers = classifers

    def classify(self, featureset):
        votes = []
        for c in self._classifers:
            v = c.classify(featureset)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


def run2():
    classifiers = run()
    voted_classifier = VoteClassifier(classifiers)


def run():
    """
    Origin Naive Bayes Algo accuracy:  67.0
    MultinomialNB Naive Bayes Algo accuracy:  72.0
    :return: 
    """
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            tup = (list(movie_reviews.words(fileid)), category)
            documents.append(tup)

    random.shuffle(documents)
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    # print(all_words.most_common(15))

    word_features = list(all_words.keys())[:3000]

    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    training_set = featuresets[:1900]
    testing_set = featuresets[1900:]

    clf = nltk.NaiveBayesClassifier.train(training_set)
    print('Origin Naive Bayes Algo accuracy: ', nltk.classify.accuracy(clf, testing_set) * 100)

    mnb_clf = SklearnClassifier(MultinomialNB())
    mnb_clf.train(training_set)
    print('MultinomialNB Algo accuracy: ', nltk.classify.accuracy(mnb_clf, testing_set) * 100)

    sgd_clf = SklearnClassifier(SGDClassifier())
    sgd_clf.train(training_set)
    print('SGDClassifier Algo accuracy: ', nltk.classify.accuracy(sgd_clf, testing_set) * 100)

    return clf, mnb_clf, sgd_clf


def main():
    # run()
    # run2()
    plot_twitter_trend()


if __name__ == '__main__':
    main()
