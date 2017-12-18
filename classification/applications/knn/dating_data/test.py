'''
    test dating dataset with knn algorithm
'''

from numpy import array
from sys import path
from os import getcwd

path.insert(0, getcwd() + '/utils/')
path.insert(0, getcwd() + '/datasets/')
path.insert(0, getcwd() + '/classification/knn/')

from process_data import process_data
from knn import classify


# import dataset, normalized dataset and class labels for dataset
dset, normalizing, labeling = process_data('datingTestSet.txt')

# normalized dataset, ranges, minimum values
# and maximum values from dataset
norm_dset, ranges, min_vals, max_vals = normalizing

# label indices to match labels for sample in dataset
# against class labels key and class labels key
label_indices, labels_key = labeling

# use 10 percent of training data as test data
ho_ratio = 0.10

# m is number of samples in dataset
m = norm_dset.shape[0]

# number of test samples
num_tests = int(m * ho_ratio)

# loop over all test samples and compare known labels versus alogrithm
# classification and print out error rate
error_count = 0.0
for i in range(num_tests):
    # normalize test sample
    norm_test = (dset[i, :] - min_vals) / ranges

    # classify test sample
    classification = classify(norm_test, norm_dset[num_tests:m, :],
                                  label_indices[num_tests:], 3)

    print('classifier answer: {}, real answer: {}'.format(
        labels_key[classification], labels_key[label_indices[i]]))

    # compare known label to classifier label
    if labels_key[classification] != labels_key[label_indices[i]]:
        error_count += 1.0

print('total error rate: {}'.format(error_count / float(num_tests)))
