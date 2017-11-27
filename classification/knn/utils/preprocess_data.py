from os import chdir, getcwd
from normalize import normalize
from to_array import to_array
from categorize_labels import categorize_labels


def preprocess_data(directory, filename):
    # save cwd and change cwd
    saved_cwd = getcwd()
    chdir(directory)

    # construct array from tab delimited file
    data, labels_list = to_array(filename)

    # create mapping of labels associated with dataset and key to index labels
    label_indices, labels = categorize_labels(labels_list)

    # normalize data
    normalized_data, ranges, min_vals, max_vals = normalize(data)

    # restore saved cwd
    chdir(saved_cwd)

    return normalized_data, labels, label_indices


if __name__ == '__main__':
    n_data, labels, label_indices = preprocess_data('dating_data', 'datingTestSet.txt')
    print(labels)