'''
    test script for adaboost algorithm
'''

from os import getcwd
from sys import path
from numpy import mat, matrix, ones

path.insert(0, getcwd() + '/classification/adaboost/')
from adaboost import build_stump


def simple_dataset():
    '''
        returns simple dataset and labels
    '''

    dset_mat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])

    labels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return dset_mat, labels

def test():
    '''
        run test script
    '''
    d_mat, labels = simple_dataset()

    D = mat(ones((5, 1)) / 5)

    build_stump(d_mat, labels, D)
