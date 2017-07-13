#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/13 17:45
"""

"""
https://www.youtube.com/watch?v=JeamFbHhmDo&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=51
"""

import pickle
import random
from collections import Counter

import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from awesomeml.pth import FILE_PATH

pickle_file = '{}/pos_neg.pickle'.format(FILE_PATH)
pos_file = '{}/pos.txt'.format(FILE_PATH)
neg_file = '{}/neg.txt'.format(FILE_PATH)

hm_lines = 10000000
lemmatizer = WordNetLemmatizer()

with open(pickle_file, 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)


def run2():
    """
    Epoch 50 completed out of 50, loss: 0.357584
    Accuracy: 0.544090
    :return: 
    """
    train_neural_network()


n_nodes_l1 = 50
n_nodes_l2 = 50
n_nodes_l3 = 50

n_classes = 2
batch_size = 100

hm_epochs = 50

x = tf.placeholder('float', [None, 423])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layer1 = {
        'weights': tf.Variable(tf.random_normal([423, n_nodes_l1])),
        'biases': tf.Variable(tf.random_normal([n_nodes_l1])),
    }

    hidden_layer2 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_l1, n_nodes_l2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_l2])),
    }

    hidden_layer3 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_l2, n_nodes_l3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_l3])),
    }

    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_l3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes])),
    }

    l1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer2['weights']), hidden_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer3['weights']), hidden_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network():
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                i += batch_size

            print('Epoch {} completed out of {}, loss: {:0.6f}'.format(epoch + 1, hm_epochs, c))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: {:0.6f}'.format(accuracy.eval({x: test_x, y: test_y})))


def run():
    train_x, train_y, test_x, test_y = create_featuresets_and_labels(pos_file, neg_file)
    with open(pickle_file, 'wb') as fw:
        pickle.dump([train_x, train_y, test_x, test_y], fw)


def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r', encoding='utf-8') as f:
            contents = f.readlines()
            for line in contents[:hm_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    # print(w_counts.most_common(3))

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r', encoding='utf-8') as f:
        contents = f.readlines()
        for line in contents[:hm_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_featuresets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos_file, lexicon, [1, 0])
    features += sample_handling(neg_file, lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


def main():
    # run()
    run2()


if __name__ == '__main__':
    main()
