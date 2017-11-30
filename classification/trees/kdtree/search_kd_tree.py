'''
    Search k-D tree by using distances
'''


from sys import path
from numpy import where
path.insert(0, '/media/gtron/files/ml/ml/utils')
from path_setter import insert_paths
insert_paths()


from knn import classify
from kd_tree import kd_tree, nn_search_tree
from process_data import process_data


def test():
    '''
        Test kdtree searching algorithm
    '''
    data, normalizing, labeling = process_data('datingTestSet.txt')
    normalized_dataset, ranges, min_vals, max_vals = normalizing
    label_indices, labels = labeling

    # set size of test set
    ho_ratio = 0.1
    m = data.shape[0]
    test_dataset_size = int(m * ho_ratio)

    training_set = normalized_dataset[test_dataset_size:]
    training_set_label_indices = label_indices[test_dataset_size:]

    tree = kd_tree(training_set)

    error_count = 0
    for i in range(test_dataset_size):
        # find nearest neighbor
        normalized_test_node = (data[:test_dataset_size][i] - min_vals) / ranges
        normalized_test_node_index = label_indices[:test_dataset_size][i]

        search_results = nn_search_tree(tree, normalized_test_node)

        kd_label = labels[training_set_label_indices[where(((training_set[:, 0] == search_results[0]) &
                                                            (training_set[:, 1] == search_results[1]) &
                                                            (training_set[:, 2] == search_results[2])))[0]]][0]

        if kd_label is not labels[normalized_test_node_index]:
            error_count += 1
    return error_count, test_dataset_size

error_count, test_dataset_size = test()
print(error_count / test_dataset_size)
