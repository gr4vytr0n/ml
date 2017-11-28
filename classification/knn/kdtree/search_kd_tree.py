from sys import path
path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn/utils')
path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn')

from kd_tree import kd_tree, knn_search_tree
from preprocess_data import preprocess_data

data, normalizing, labeling = preprocess_data(
    'dating_data', 'datingTestSet.txt')
normalized_dataset, ranges, min_vals, max_vals = normalizing
label_indices, labels = labeling

# set size of test set
ho_ratio = 0.1
m = data.shape[0]
test_dataset_size = int(m * ho_ratio)

training_set = normalized_dataset[test_dataset_size:]

tree = kd_tree(training_set)

count_same = 0
for i in range(test_dataset_size):
    # find 3 nearest neighbors
    normalized_test_node = (data[:test_dataset_size][i] - min_vals) / ranges
    search_results = knn_search_tree(tree, normalized_test_node)

    from numpy import where
    kd_label = labels[label_indices[where(((training_set[:, 0] == search_results[0]) &
                                    (training_set[:, 1] == search_results[1]) &
                                    (training_set[:, 2] == search_results[2])))[0]]][0]
    print(kd_label)

    from knn import classify0

    result = classify0(normalized_test_node, training_set,
                    label_indices[test_dataset_size:], 3)

    print(labels[result])
    print(labels[label_indices[i]])

    if labels[result] == kd_label:
        count_same += 1
print(count_same/100)