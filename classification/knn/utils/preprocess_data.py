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
    labeling = categorize_labels(labels_list)

    # normalize data
    normalizing = normalize(data)

    # restore saved cwd
    chdir(saved_cwd)

    return data, normalizing, labeling

    # data, normalizing, labeling = preprocess_data('directory', 'file.txt')

    # n_data, ranges, min_val, max_val = normalizing

    # label_indices, labels = labeling
