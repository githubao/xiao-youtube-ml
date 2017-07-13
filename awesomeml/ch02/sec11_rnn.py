#!/usr/bin/env python
# encoding: utf-8

"""
@description: rnn

@author: BaoQiang
@time: 2017/7/13 20:25
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from awesomeml.pth import FILE_PATH
import numpy as np

mnist = input_data.read_data_sets(FILE_PATH, one_hot=True)

n_nodes_l1 = 50
n_nodes_l2 = 50
n_nodes_l3 = 50

n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28

rnn_size = 128
hm_epochs = 3

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def run():
    """
    Epoch 3 completed out of 3, loss: 0.050186
    Accuracy: 0.975700
    :return: 
    """
    train_neural_network()


def recurrent_neural_network(X):
    layer = {
        'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes])),
    }

    X = tf.transpose(X, [1, 0, 2])
    X = tf.reshape(X, [-1, chunk_size])
    X = tf.split(X, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    output, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)

    output = tf.matmul(output[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network():
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / batch_size)):
                x_batch, y_batch = mnist.train.next_batch(batch_size)

                x_batch = x_batch.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
                epoch_loss += c
            print('Epoch {} completed out of {}, loss: {:0.6f}'.format(epoch + 1, hm_epochs, c))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: {:0.6f}'.format(
            accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels})))


def main():
    run()


if __name__ == '__main__':
    main()
