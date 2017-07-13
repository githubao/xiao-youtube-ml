#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/13 16:23
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from awesomeml.pth import FILE_PATH
import numpy as np

mnist = input_data.read_data_sets(FILE_PATH, one_hot=True)

n_nodes_l1 = 50
n_nodes_l2 = 50
n_nodes_l3 = 50

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def run():
    """
    Epoch 10 completed out of 10, loss: 6.102884
    Accuracy: 0.850200
    
    :return: 
    """
    train_neural_network()


def neural_network_model(data):
    hidden_layer1 = {
        'weights': tf.Variable(tf.random_normal([784, n_nodes_l1])),
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

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / batch_size)):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
                epoch_loss += c
            print('Epoch {} completed out of {}, loss: {:0.6f}'.format(epoch + 1, hm_epochs, c))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: {:0.6f}'.format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))


def tmp():
    # a = tf.placeholder(dtype=tf.float32)
    # b = tf.placeholder(dtype=tf.float32)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     result = tf.add(a, b)
    #     print(sess.run(result, feed_dict={a: 32, b: 10}))

    a = np.array([[1, 2], [3, 4]])
    print(np.sum(a, axis=0))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = tf.reduce_sum(a, axis=0)
        print(sess.run(result))


def main():
    # tmp()
    run()


if __name__ == '__main__':
    main()
