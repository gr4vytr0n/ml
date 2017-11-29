'''
    kNN (k-Nearest Neighbor) classification algorithm
'''

from operator import itemgetter
from numpy import tile


def classify(test_set, dataset, labels, k):
    '''
        Classify k-Nearest neighbors
    '''
    dataset_size = dataset.shape[0]
    diff_mat = tile(test_set, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat**2
    squared_distances = sq_diff_mat.sum(axis=1)
    distances = squared_distances**0.5
    sorted_distancess = distances.argsort()

    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distancess[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(
        class_count.items(), key=itemgetter(1), reverse=True)
    # print(sorted_class_count)
    return sorted_class_count[0][0]
