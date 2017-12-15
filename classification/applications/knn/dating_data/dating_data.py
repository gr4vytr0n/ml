'''
    dating data classifier application
    takes user input for three features to create a sample
    that will be classified
'''

from numpy import array
from sys import path
from os import getcwd

path.insert(0, '/media/gtron/files/ml/ml/utils/')
path.insert(0, '/media/gtron/files/ml/ml/datasets/')
path.insert(0, '/media/gtron/files/ml/ml/classification/knn/')

from process_data import process_data
from knn import classify


dset, normalizing, labeling = process_data('datingTestSet.txt')
norm_set, ranges, min_vals, max_vals = normalizing
label_indices, label_keys = labeling

# classes key
class_names = ['not at all', 'in small doses', 'in large doses']

# user input -- sample for testing
gaming = float(input('percent time playing video games?'))
flyerMiles = float(input('frequent flyer miles earned each year?'))
iceCream = float(input('liters of ice cream consumed per year?'))
user_input = array([flyerMiles, gaming, iceCream])

# normalize test sample
norm_test = (user_input - min_vals) / ranges

# classify user input
classifications = classify(norm_test, norm_set, label_indices, 3)
print('You will probably like this person: ', class_names[classifications - 1])
