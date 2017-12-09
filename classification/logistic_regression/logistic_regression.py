'''
    logistic regression
'''
from math import exp
from numpy import mat, shape, ones, exp


def load_dataset(filename):
    ''' load data '''
    dataset = []
    labels = []

    with open(filename) as file:
        for line in file.readlines():
            line_list = line.strip().split()
            dataset.append([1.0, float(line_list[0]), float(line_list[1])])
            labels.append(int(line_list[2]))
    
    return dataset, labels


def sigmoid(in_x):
    ''' sigmoid function '''
    return 1.0 / (1 + exp(-in_x))


def gradient_ascent(d_set, l_set):
    ''' gradient ascent '''
    dataset_matrix = mat(d_set)
    labels_matrix = mat(l_set).transpose()

    m, n = shape(dataset_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))

    for _ in range(max_cycles):
        h = sigmoid(dataset_matrix * weights)
        error = (labels_matrix - h)
        weights = weights + alpha * dataset_matrix.transpose() * error

    return weights

if __name__ == '__main__':
    dataset, labels = load_dataset('/media/gtron/files/ml/ml/classification/logistic_regression/testSet.txt')
    
    print(gradient_ascent(dataset, labels))
