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

# --------------------- k Nearest Neigbors

# classify dating data
# use test set
sys.path.insert(0, CWD + '/classification/applications/knn/dating_data/')
from dating_data_test import test as knn_dating_data_test

# get user input for dating data test set
sys.path.insert(0, CWD + '/classification/applications/knn/dating_data/')
from dating_data import test as knn_dating_data_user_input

# classify handwriting set
sys.path.insert(0, CWD + '/classification/applications/knn/handwriting_data/')
from handwriting_recognition import test as knn_hw_test

# --------------------- decision trees

# run test script
sys.path.insert(0, CWD + '/classification/applications/trees/')
from decision_trees_test import test as d_trees_test

# contact lenses classification test
sys.path.insert(0, CWD + '/classification/applications/trees/')
from contact_lenses import test as lenses_test

# --------------------- naive bayes

# spam classification test
sys.path.insert(0, CWD + '/classification/applications/bayes/')
from spam import test as spam_test

# local attitudes classification test
sys.path.insert(0, CWD + '/classification/applications/bayes/')
from local_attitudes import test as local_attitudes_test

# --------------------- logistic regression

# plot best fit test
sys.path.insert(0, CWD + '/classification/applications/logistic_regression/')
from plot_best_fit import test as best_fit_test

# run test script
sys.path.insert(0, CWD + '/classification/applications/logistic_regression/')
from horse_colic import test as horse_colic_test

