from to_array import to_array
from categorize_labels import categorize_labels


def preprocess_data(filename):
    data, labels_list = to_array(filename)
    label_indices, labels = categorize_labels(labels_list)

    return data, labels, label_indices
