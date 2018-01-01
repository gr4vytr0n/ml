'''
    run tests using SVM
'''

from numpy import mat 
from os import getcwd, chdir
from sys import path
path.insert(0, getcwd() + '/classification/support_vector_machines/')
from support_vector_machines import *


def test_rbf(k1=1.3):
    ''' radial bias function kernel '''
    save_cwd = getcwd()
    chdir(getcwd() + '/datasets/svm/')
    dataset, labels = load_dset('testSetRBF.txt')

    b, alphas = platt_smo(dataset, labels, 200, 0.0001, 10000, ('rbf', k1))
    dset_mat = mat(dataset)
    lbls_mat = mat(labels).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    sVs = dset_mat[sv_ind]
    labelSV = lbls_mat[sv_ind]
    print('there are {} support vectors'.format(shape(sVs)[0]))
    m, n = shape(dset_mat)
    error_cnt = 0
    for i in range(m):
        kernel_eval = kernel_trans(sVs, dset_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(labelSV, alphas[sv_ind]) + b
        if sign(predict) != sign(labels[i]):
            error_cnt += 1
    print('the training error rate is: {}'.format(float(error_cnt) / m))
    dataset, labels = load_dset('testSetRBF2.txt')
    error_cnt = 0
    dset_mat = mat(dataset)
    lbls_mat = mat(labels).transpose()
    m, n = shape(dset_mat)
    for i in range(m):
        kernel_eval = kernel_trans(sVs, dset_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(labelSV, alphas[sv_ind]) + b
        if sign(predict) != sign(labels[i]):
            error_cnt += 1
        print('the test error rate is: {}'.format(float(error_cnt) / m))

    chdir(save_cwd)

def test():
    ''' run script '''
    
    # test rbf kernel
    test_rbf()
    
    # dataset, labels = load_dset(filename)

    # b, alphas = simple_smo(dataset, labels, 0.6, 0.001, 40)
    # b, alphas = simple_smo(dataset, labels, 0.6, 0.001, 40)

    # ws = calc_Ws(alphas, dataset, labels)
    # dmat = mat(dataset)
    # for i in range(10):
    #     print(dmat[i] * mat(ws) + b)
    #     print(labels[i])

    # print(b)
    # print(alphas[alphas > 0])
    # print(shape(alphas[alphas > 0]))
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         print(dataset[i], labels[i])
