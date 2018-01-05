'''
    use pandas to process data
    and run scikit-learn version of knn on data
'''

from os import chdir, getcwd, listdir
from sys import path

path.insert(0, getcwd() + '/using_libs/classification/knn/')

from knn import knn_classify as classify

import pandas as pd
import numpy as np


def load_dataset(path, filename):
    '''
        use pandas to load csv file
    '''

    save_cwd = getcwd()
    chdir(path)

    df = pd.read_csv(filename, encoding='CP1252', header=None)

    chdir(save_cwd)

    return df


def build_dataset(dataset_path):
    '''
        build list of dataframes
    '''

    dataset = []
    filenames = listdir(dataset_path)
    for filename in filenames:
        dataset.append(load_dataset(dataset_path, filename))

    return dataset, filenames


def to_vectors(data):
    '''
        convert arrays of handwriting into vectors of 1s and 0s
    '''

    vectors = np.zeros((len(data), 1024))
    for idx, value in enumerate(data):
        lines = value.values
        vector = np.zeros((1, 1024))
        for i in range(32):
            line = lines[i][0]
            for j in range(32):
                vector[0, 32 * i + j] = int(line[j])
        vectors[idx, :] = vector

    return vectors

def labels_from_filenames(filenames):
    '''
        extract label from file name
    '''

    return [int(f.split('.')[0].split('_')[0]) for f in filenames]


def test():
    '''
        run test script
    '''

    train_dir = getcwd() + '/datasets/hw/trainingDigits/'
    train_dataset, train_filenames = build_dataset(train_dir)
    train_labels = labels_from_filenames(train_filenames)
    train_vectors = to_vectors(train_dataset)

    test_dir = getcwd() + '/datasets/hw/testDigits/'
    test_dataset, test_filenames = build_dataset(test_dir)
    test_labels = labels_from_filenames(test_filenames)
    test_vectors = to_vectors(test_dataset)

    n_neighbors = classify(train_vectors, test_vectors, train_labels)

    err_count = 0
    for idx, prediction in enumerate(n_neighbors):
        if prediction != test_labels[idx]:
            err_count += 1
    
    print('error rate: {}'.format(err_count / float(len(test_vectors))))
