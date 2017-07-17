#!/usr/bin/env python
# encoding: utf-8

"""
@description:  猫狗识别

@author: BaoQiang
@time: 2017/7/17 13:53
"""

import os
from random import shuffle

import cv2
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm
import matplotlib.pyplot as plt

from awesomeml.pth import FILE_PATH

TRAIN_DIR = '{}/dogs_vs_cats/train/train'.format(FILE_PATH)
TEST_DIR = '{}/dogs_vs_cats/test/test'.format(FILE_PATH)
IMAGE_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        train_data.append([np.array(img), np.array(label)])

    shuffle(train_data)
    # np.save('train_data.npy',train_data)
    return train_data


def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        test_data.append([np.array(img), img_num])

    shuffle(test_data)
    # np.save('test_data.npy',test_data)
    return test_data


def run():
    train_data = create_train_data()
    # train_data = np.load('train_data.npy')

    convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded from file')

    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = [i[1] for i in train]

    test_X = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': y}, n_epoch=3, validation_set={{'input': test_X}, {'targets': test_y}},
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

    return model


def test():
    model = run()
    model = model.load(MODEL_NAME)

    test_data = process_test_data()
    fig = plt.figure()

    for num, data in enumerate(test_data[:12]):
        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        data = img_data.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xasis().set_vasible(True)
        y.axes.get_yasis().set_vasible(True)

    plt.show()

    with open('submission-file.csv', 'w', encoding='utf-8') as fw:
        fw.write('id,label\n')
        for data in tqdm(test_data):
            img_num = data[1]
            img_data = data[0]

            data = img_data.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

            model_out = model.predict([data])[0]
            fw.write('{},{}\n'.format(img_num, model_out[1]))


def main():
    run()


if __name__ == '__main__':
    main()
