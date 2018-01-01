'''
    ml.py

    execute project specific functions
'''

import os
import sys

# cwd
CWD = os.getcwd()

'''
    classification
'''

# k Nearest Neigbors

# classify dating data
# use test set
sys.path.insert(0, CWD + '/classification/applications/knn/dating_data/')
from dating_data_test import test

def knn_dating_data_test():
    test()

# get user input for dating data test set
sys.path.insert(0, CWD + '/classification/applications/knn/dating_data/')
from dating_data import test

def knn_dating_data_user_input():
    test()

# classify handwriting set
sys.path.insert(0, CWD + '/classification/applications/knn/handwriting_data/')
from handwriting_recognition import test

def knn_hw_test():
    test()

# decision trees

# run test script
sys.path.insert(0, CWD + '/classification/applications/trees/')
from decision_trees_test import test

def d_trees_test():
    test()
