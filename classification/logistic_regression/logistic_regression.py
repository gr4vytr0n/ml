'''
    logistic regression
'''


from math import exp
from random import uniform
from os import getcwd, chdir

import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    ''' load dataset '''
    save_cwd = getcwd()
    chdir(save_cwd + '/datasets/')

    with open(filename) as file:
        lines = [line.strip().split() for line in file.readlines()]
        dataset = [[1.0, float(line[0]), float(line[1])] for line in lines]
        labels = [int(line[2]) for line in lines]

    chdir(save_cwd)

    return dataset, labels


def sigmoid(in_x):
    ''' sigmoid function '''
    return np.exp(-np.logaddexp(0, -in_x))


def gradient_ascent(d_set, l_set):
    ''' gradient ascent '''
    dataset_matrix = np.mat(d_set)
    labels_matrix = np.mat(l_set).transpose()

    m, n = np.shape(dataset_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))

    for _ in range(max_cycles):
        h = sigmoid(dataset_matrix * weights)
        error = (labels_matrix - h)
        weights = weights + alpha * dataset_matrix.transpose() * error

    return weights


def stochastic_gradient_ascent(d_set, l_set, num_iters=150):
    ''' stochastic gradient ascent '''
    m, n = np.shape(d_set)
    weights = np.ones(n)

    for j in range(num_iters):
        d_idx = list(range(m))

        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_idx = int(uniform(0, len(d_idx)))
            h = sigmoid(sum(d_set[rand_idx] * weights))
            error = l_set[rand_idx] - h
            weights = weights + alpha * error * d_set[rand_idx]
            del d_idx[rand_idx]

    return weights


def plot_best_fit(wei):
    ''' ploy the decision boundary '''
    if isinstance(wei, np.ndarray):
        plt_weights = wei
    else:
        plt_weights = wei.getA()

    d_set, l_set = load_dataset('testSet.txt')
    dataset_array = np.array(d_set)
    n = np.shape(dataset_array)[0]
    x_coord_1 = []
    y_coord_1 = []
    x_coord_2 = []
    y_coord_2 = []

    for i in range(n):
        if int(l_set[i]) == 1:
            x_coord_1.append(dataset_array[i, 1])
            y_coord_1.append(dataset_array[i, 2])
        else:
            x_coord_2.append(dataset_array[i, 1])
            y_coord_2.append(dataset_array[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_coord_1, y_coord_1, s=30, c='red', marker='s')
    ax.scatter(x_coord_2, y_coord_2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-plt_weights[0] - plt_weights[1] * x) / plt_weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataset, labels = load_dataset('testSet.txt')

    weights = stochastic_gradient_ascent(np.array(dataset), labels)

    plot_best_fit(weights)
