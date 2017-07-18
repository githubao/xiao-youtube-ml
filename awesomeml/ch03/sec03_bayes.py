#!/usr/bin/env python
# encoding: utf-8

"""
@description: 朴素贝叶斯分类

@author: BaoQiang
@time: 2017/7/18 13:08
"""

import nltk
import random
from nltk.corpus import movie_reviews
from awesomeml.pth import FILE_PATH

pickle_file = '{}/naive_bayes.pickle'.format(FILE_PATH)

import pickle


# nltk.download()


def run():
    """
    Naive Bayes Algo accuracy:  73.0
    :return: 
    """

    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            tup = (list(movie_reviews.words(fileid)), category)
            documents.append(tup)

    # documents = [(list(movie_reviews.words(fileid)), category)
    #              for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)
    #              ]

    random.shuffle(documents)
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    print(all_words.most_common(15))

    word_features = list(all_words.keys())[:3000]

    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    print(find_features(movie_reviews.words('neg/cv000_29416.txt')))
    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    training_set = featuresets[:1900]
    testing_set = featuresets[1900:]

    clf = nltk.NaiveBayesClassifier.train(training_set)
    print('Naive Bayes Algo accuracy: ', nltk.classify.accuracy(clf, testing_set) * 100)
    clf.show_most_informative_features(15)

    with open(pickle_file,'wb') as fw:
        pickle.dump(clf,fw)

def rerun():
    with open(pickle_file, 'rb') as f:
        clf = pickle.load(f)

    clf.show_most_informative_features(10)

def main():
    run()
    rerun()

if __name__ == '__main__':
    main()
