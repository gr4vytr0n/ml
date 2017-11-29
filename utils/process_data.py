'''
    Process data for use
'''

from os import chdir, getcwd
from normalize import normalize
from to_array import to_array
from categorize_labels import categorize_labels


def process_data(filename):
    '''
        Read data from file and prepare for processing
    '''

    # save cwd and change cwd
    saved_cwd = getcwd()
    chdir('datasets')

    # construct array from tab delimited file
    data, labels_list = to_array(filename)

    # create mapping of labels associated with dataset and key to index labels
    labeling = categorize_labels(labels_list)

    # normalize data
    normalizing = normalize(data)

    # restore saved cwd
    chdir(saved_cwd)

    return data, normalizing, labeling
