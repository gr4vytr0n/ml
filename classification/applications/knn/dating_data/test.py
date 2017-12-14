'''
    test dating dataset with knn algorithm
'''

from numpy import array
from sys import path
from os import getcwd

path.insert(0, '/media/gtron/files/ml/ml/utils/')
path.insert(0, '/media/gtron/files/ml/ml/datasets/')
path.insert(0, '/media/gtron/files/ml/ml/classification/knn/')

from process_data import process_data
from knn import classify


def test():
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

    # number of test vectors
    num_test_vectors = int(m * ho_ratio)

    # loop over all test vectors and compare known label versus alogrithm
    # classification and print out error rate
    error_count = 0.0
    for i in range(num_test_vectors):
        # normalize test vector
        norm_test = (dset[i, :] - min_vals) / ranges

        # classify test vector
        classifier_results = classify(norm_test, norm_dset[num_test_vectors:m, :],
                                      label_indices[num_test_vectors:], 3)

        print('classifier answer: {}, real answer: {}'.format(
            labels_key[classifier_results], labels_key[label_indices[i]]))

        # compare known label to classifier label
        if labels_key[classifier_results] != labels_key[label_indices[i]]:
            error_count += 1.0

    print('total error rate: {}'.format(error_count / float(num_test_vectors)))


test()
