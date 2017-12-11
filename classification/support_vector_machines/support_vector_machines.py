'''
    support vector machines
'''
from random import uniform
from numpy import mat, shape, multiply


'''
    SMO (sequential minimal optimization)
'''
def simple_smo(dataset, labels, const, tol, max_iter):
    '''
        simplified version of smo algorithm
    '''
    dset_mat = mat(dataset)
    labels_mat = mat(labels).transpose()

    b = 0
    m, n = shape(dset_mat)
    alphas = mat(zeros((m, 1)))
    iter = 0

    while (iter < max_iter):
        alpha_pairs_changed = 0

        for i in range(m):
            fXi = float(multiply(alphas, labels_mat).T * (dset_mat * dset_mat[i, :].T)) + b
            Ei = fXi - float(labels_mat[i])
            if ((labels_mat[i] * Ei < -tol) and (alphas[i] < const)) or \
               ((labels_mat[i] * Ei > tol) and (alphas[i] > const)):
               j = random_int(i, m)
               fXj = float(multiply(alphas, labels_mat).T * (dset_mat * dset_mat[j, :].T)) + b
               Ej = fXj - float(labels_mat[j])


def load_dset(filename):
    ''' load dataset '''
    with open(filename) as f_r:
        dset_array = []
        labels_array = []

        for line in f_r.readlines():
            line_array = line.strip().split('\t')
            dset_array.append([float(line_array[0]), float(line_array[1])])
            labels_array.append(float(line_array[2]))

        return dset_array, labels_array


def random_int(alpha_idx, num_alphas):
    ''' random integer from range '''
    rand_int = alpha_idx

    while (rand_int == alpha_idx):
        rand_int = int(uniform(0, num_alphas))

    return rand_int


def clip_alpha(alpha, h_val, l_val):
    ''' clip values if the values get too big '''
    if alpha > h_val:
        alpha = h_val

    if l_val > alpha:
        alpha = l_val

    return alpha


def main(filename):
    ''' run script '''
    dataset = load_dset(filename)
    print(dataset)


main('/media/gtron/files/ml/ml/classification/' +
     'support_vector_machines/testSet.txt')
