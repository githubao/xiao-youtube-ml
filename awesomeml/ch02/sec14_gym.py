#!/usr/bin/env python
# encoding: utf-8

"""
@description: openai 游戏 https://gym.openai.com

@author: BaoQiang
@time: 2017/7/14 20:06
"""

import gym
import numpy as np
import tflearn
from tflearn import fully_connected, input_data, dropout
from tflearn.layers.estimator import regression
from statistics import mean, median
import random
from collections import Counter

from awesomeml.pth import FILE_PATH

LR = 1e-3

env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
initial_games = 10000
score_requirement = 50


# goal_steps = 50
# score_requirement = 0
# initial_games = 10


def some_random_games():
    for episode in range(20):
        env.reset()

        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('{}/saved.npy'.format(FILE_PATH), training_data_save)
    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))

    print(Counter(accepted_scores))
    return training_data


def run():
    initial_population()


def main():
    run()


if __name__ == '__main__':
    main()
