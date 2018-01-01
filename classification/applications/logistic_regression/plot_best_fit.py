'''
    plot best fit using gradient ascent (stochastic)
'''

from os import getcwd, chdir

from numpy import array

from  sys import path
path.insert(0, getcwd() + '/classification/logistic_regression/')
from logistic_regression import *

def test():
    '''
        run test script
    '''
    dataset, labels = load_dataset('testSet.txt')

    weights = stochastic_gradient_ascent(array(dataset), labels)

    plot_best_fit(weights)