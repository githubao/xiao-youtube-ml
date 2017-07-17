#!/usr/bin/env python
# encoding: utf-8

"""
@description: 实现自己的一个卷积神经网络

@author: BaoQiang
@time: 2017/7/17 15:29
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import dicom
import cv2
import math

IMG_SIZE_PX = 50
SLICE_COUNT = 20
data_dir = '../input/sample_images'

n_classes = 2
hm_epochs = 10

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize=False):
    label = labels_df.get_value(patient, 'cancer')
    path = os.path.join(data_dir, patient)
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    # print(len(slices), label)
    print(len(slices), slices[0].pixel_array.shape)
    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (img_px_size, img_px_size)) for each_slice in slices]
    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chucks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
    # resize the data
    if len(new_slices) == hm_slices - 1:
        new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices + 2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val
    if len(new_slices) == hm_slices + 1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val
    print(len(new_slices))
    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices[:12]):
            y = fig.add_subplot(3, 4, num + 1)
            y.imshow(each_slice.pixel_array, cmap='gray')
        plt.show()

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])

    return np.array(new_slices), label


def mean(a):
    return sum(a) / len(a)


def chucks(l, n):
    for i in range(0, len(l), n):
        yield l[i, i + n]


def load_data():
    patients = os.listdir(data_dir)
    labels_df = pd.read_csv('../input/stage_labels.csv', index_col=0)
    much_data = []
    for num, patient in enumerate(patients):
        if num % 100 == 0:
            print(num)
        try:
            img_data, label = process_data(patient, labels_df, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
            much_data.append([img_data, label])
        except KeyError as e:
            print('This is unlabel data')

    # np.save('muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT), much_data)
    return much_data


def train_neural_network():
    much_data = load_data()

    train_data = much_data[:-100]
    test_data = much_data[-100:]

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for data in train_data:
                _, c = sess.run([optimizer, cost], feed_dict={x: data[0], y: data[1]})
                epoch_loss += c
            print('Epoch {} completed out of {}, loss: {:0.6f}'.format(epoch + 1, hm_epochs, c))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: {:0.6f}'.format(accuracy.eval({x: [i[0] for i in test_data], y: [i[1] for i in test_data]})))


def convolutional_neural_network(data):
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
        'W_fc': tf.Variable(tf.random_normal([50000, 1024])),
        'output': tf.Variable(tf.random_normal([1024, n_classes])),
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'output': tf.Variable(tf.random_normal([n_classes])),
    }

    x = tf.reshape(data, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 50000])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    fc = tf.nn.dropout(fc, 0.8)

    output = tf.matmul(fc, weights['output']) + biases['output']

    return output


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def main():
    pass


if __name__ == '__main__':
    main()
