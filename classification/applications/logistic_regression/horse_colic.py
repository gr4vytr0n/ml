'''
    use logistic regression to find out if a horse
    with colic will live or die
'''

from os import getcwd, chdir

from numpy import sum, array

from  sys import path
path.insert(0, getcwd() + '/classification/logistic_regression/')
from logistic_regression import *

def classifier(in_x, weights):
    ''' classify vector '''
    prob = sigmoid(sum(in_x * weights))

    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    '''
        run script to classify
    '''
    save_cwd = getcwd()
    chdir(save_cwd + '/datasets/horse_colic/')

    training_set = []
    training_labels = []

    with open('horseColicTraining.txt') as train_file:
        for line in train_file.readlines():
            curr_line = line.strip().split('\t')
            line_array = []

            for i in range(21):
                line_array.append(float(curr_line[i]))

            training_set.append(line_array)
            training_labels.append(float(curr_line[21]))

    train_weights = stochastic_gradient_ascent(
        array(training_set), training_labels, 500)

    error_count = 0
    num_test_vectors = 0.0

    with open('horseColicTest.txt') as test_file:
        for line in test_file.readlines():
            num_test_vectors += 1.0

            curr_line = line.strip().split('\t')
            line_array = []

            for i in range(21):
                line_array.append(float(curr_line[i]))

            if int(classifier(array(line_array), train_weights)) != int(curr_line[21]):
                error_count += 1

    error_rate = float(error_count) / num_test_vectors

    print('error rate: {}'.format(error_rate))

    chdir(save_cwd)

    return error_rate

def test(num):
    ''' run colic_test() num number of times '''
    error_sum = 0.0

    for k in range(num):
        error_sum += colic_test()

    print('after {} iterations the average error rate is: {}'.format(num, (error_sum / float(num))))
