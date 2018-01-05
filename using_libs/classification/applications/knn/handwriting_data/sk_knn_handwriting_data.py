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

from sklearn import preprocessing


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

    return dataset


def to_vector(data):
    '''
        convert arrays of handwriting into vectors of 1s and 0s
    '''

    vectors = np.zeros((len(data), 1024))
    for idx, value in enumerate(data):
        lines = value.values
        vector = np.zeros((1, 1024))
        for i in range(32):
            line = lines[i]
            for j in range(32):
                vector[0, 32 * i + j] = int(line[j])
        vectors[idx, :] = vector

    return vectors


def test():
    '''
        run test script
    '''

    train_dir = getcwd() + '/datasets/hw/trainingDigits/'
    train_dataset = build_dataset(train_dir)

    test_dir = getcwd() + '/datasets/hw/testDigits/'
    test_dataset = build_dataset(test_dir)

    print(to_vector(train_dataset))#[0].values)

    # dataset, labels = load_dataset()

    # le = preprocessing.LabelEncoder()
    # encoded_labels = le.fit_transform(labels)

    # n_neighbor = classify(dataset, [[1000, 0.5, 340]], encoded_labels)

    # print(le.inverse_transform(n_neighbor))
