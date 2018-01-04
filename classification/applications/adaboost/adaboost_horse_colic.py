'''
    run adaboost classifier on horse colic data set
'''

from os import chdir, getcwd
from sys import path

from numpy import mat, ones

path.insert(0, getcwd() + '/classification/adaboost/')

from adaboost import adaboost_trainer_ds, adaboost_classify_ds


def load_dataset(filename):
    '''
        load horse colic data set
    '''

    save_cwd = getcwd()
    chdir(getcwd() + '/datasets/horse_colic/')

    num_feat = len(open(filename).readline().split('\t'))
    d_mat = []
    l_mat = []

    fr = open(filename)

    for line in fr.readlines():
        line_arr = []
        curr_line = line.strip().split('\t')

        for i in range(num_feat - 1):
            line_arr.append(float(curr_line[i]))

        d_mat.append(line_arr)
        l_mat.append(float(curr_line[-1]))

    chdir(save_cwd)

    return d_mat, l_mat


def test():
    '''
        run script test
    '''

    d_train_arr, l_train_arr = load_dataset('horseColicTraining2.txt')
    d_test_arr, l_test_arr = load_dataset('horseColicTest2.txt')

    classifier_arr = adaboost_trainer_ds(d_train_arr, l_train_arr, 10)

    prediction_10 = adaboost_classify_ds(d_test_arr, classifier_arr)

    err_arr = mat(ones((67, 1)))

    print(err_arr[prediction_10 != mat(l_test_arr).T].sum() / 67)
