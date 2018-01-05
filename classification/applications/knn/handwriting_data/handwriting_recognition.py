'''
    hand written digits classification using kNN
'''


from os import getcwd, listdir, chdir, path as os_path
from time import perf_counter
from numpy import zeros, array

from  sys import path, argv
path.insert(0, getcwd() + '/classification/knn')
from knn import *


def build_file_lists(path, dirs):
    ''' insert filenames from given directories into a list '''

    save_cwd = getcwd()
    chdir(path)

    # test at index 0, training at index 1
    filenames = [listdir(dir) for dir in dirs]

    chdir(path + dirs[0])
    test_files = []
    for number_file in filenames[0]:
        with open(number_file) as n_file:
            test_files.append(n_file.read())

    chdir(path + dirs[1])
    train_files = []
    for number_file in filenames[1]:
        with open(number_file) as n_file:
            train_files.append(n_file.read())

    chdir(save_cwd)

    return test_files, train_files, filenames


def to_vectors(data):
    ''' convert arrays of handwriting into vectors of 1s and 0s '''

    vectors = zeros((len(data), 1024))
    for idx, value in enumerate(data):
        lines = value.split('\n')
        vector = zeros((1, 1024))
        for i in range(32):
            line = lines[i]
            for j in range(32):
                vector[0, 32 * i + j] = int(line[j])
        vectors[idx, :] = vector

    return vectors


def classify_handwriting(test_set, train_set, f_names):
    ''' classifiy test vectors '''

    test_labels = [int(f.split('.')[0].split('_')[0]) for f in f_names[0]]

    train_labels = [int(f.split('.')[0].split('_')[0]) for f in f_names[1]]

    error_count = 0.0
    for test in range(len(test_set)):
        classification = classify(test_set[test], train_set, train_labels, 3)
        # print('classifier result: {}, real answer: {}'.format(classifierResult, test_labels[test]))
        if (classification != test_labels[test]):
            error_count += 1.0

    print('total errors: {}'.format(error_count))
    print('total error rate: {}'.format(error_count/float(len(test_set))))

def test():
    '''
        run script
    '''
    # create arrays of sample vectors for test and training datasets
    file_path = getcwd() + '/datasets/hw/'
    file_dirs = ['testDigits', 'trainingDigits']
    test_files, train_files, filenames = build_file_lists(file_path, file_dirs)
    test_set = to_vectors(test_files)
    train_set = to_vectors(train_files)

    #t0 = perf_counter()

    classify_handwriting(test_set, train_set, filenames)

    #t1 = perf_counter()
    #print('elapsed time: {}'.format(t1 - t0))
