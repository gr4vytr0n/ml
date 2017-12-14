'''
    kNN (k-Nearest Neighbor) classification algorithm
    ``````````````````````````````````````````````````
    * start with training dataset (w/ labels)
    * compare test dataset to all samples in training dataset
    * take k number of closest matches
    * classify test data as majority class from k nearest neighbors
'''

from operator import itemgetter
from numpy import tile


def classify(test_set, dataset, labels, k):
    '''
        Classify k-Nearest neighbors
    '''

    # number of samples in dataset
    dataset_size = dataset.shape[0]

    # get the Euclidian distance from test_set to each sample
    diff_mat = tile(test_set, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat**2
    squared_distances = sq_diff_mat.sum(axis=1)
    distances = squared_distances**0.5
    sorted_distancess = distances.argsort()

    # count occurences of classes in k nearest neighbors
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distancess[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # sort list of class occurence counts
    sorted_class_count = sorted(
        class_count.items(), key=itemgetter(1), reverse=True)

    # return majority class from k nearest neighbors
    return sorted_class_count[0][0]
