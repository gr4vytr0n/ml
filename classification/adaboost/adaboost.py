'''
    adaboost meta-alogorithm
'''

from numpy import ones, shape, mat, zeros, exp, multiply, sign, log
from math import inf


def stump_classify(d_mat, dimen, thresh_val, thresh_ineq):
    '''
        descision stump classifier
    '''

    result_arr = ones((shape(d_mat)[0], 1))

    if thresh_ineq == 'lt':
        result_arr[d_mat[:, dimen] <= thresh_val] = -1.0
    else:
        result_arr[d_mat[:, dimen] > thresh_val] = -1.0

    return result_arr


def build_stump(d_arr, labels, D):
    '''
        build stump
    '''

    d_mat = mat(d_arr)
    l_mat = mat(labels).T
    m, n = shape(d_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = mat(zeros((m, 1)))
    min_err = inf

    for i in range(n):
        range_min = d_mat[:, i].min()
        range_max = d_mat[:, i].max()

        step_size = (range_max - range_min) / num_steps

        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(d_mat, i, thresh_val, inequal)
                err_arr = mat(ones((m, 1)))
                err_arr[predicted_vals == l_mat] = 0
                weighted_err = D.T * err_arr
                #print('split: dim {}, thresh {}, thresh inequal {}, weighted err {}'.format(i, thresh_val, inequal, weighted_err))

                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal

    return best_stump, min_err, best_class_est


def adaboost_trainer_ds(d_arr, labels, num_iter=40):
    '''
        adaboost trainer
    '''

    weak_class_arr = []
    m = shape(d_arr)[0]
    D = mat(ones((m, 1)) / m)
    agg_class_est = mat(zeros((m, 1)))

    for i in range(num_iter):
        best_stump, err, class_est = build_stump(d_arr, labels, D)

        #print('D: {}'.format(D.T))

        alpha = float(0.5 * log((1.0 - err) / max(err, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)

        #print('class_est: {}'.format(class_est.T))

        expon = multiply(-1 * alpha * mat(labels).T, class_est)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        agg_class_est += alpha * class_est

        #print('agg_class_est: {}'.format(agg_class_est.T))

        agg_errs = multiply(sign(agg_class_est) != mat(labels).T, ones((m, 1)))
        err_rate = agg_errs.sum() / m

        print('total error: {}\n'.format(err_rate))

        if err_rate == 0.0:
            break

    return weak_class_arr


def adaboost_classify_ds(d_to_class, classifier_arr):
    '''
        adaboost classifier
    '''

    d_mat = mat(d_to_class)
    m = shape(d_mat)[0]
    agg_class_est = mat(zeros((m, 1)))

    for i in range(len(classifier_arr)):
        class_est = stump_classify(d_mat, classifier_arr[i]['dim'],
                                   classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est

        #print(agg_class_est)

    return sign(agg_class_est)
