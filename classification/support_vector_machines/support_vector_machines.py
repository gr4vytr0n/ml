'''
    support vector machines
'''
from random import uniform
from numpy import mat, shape, multiply, zeros, abs


'''
    SMO (sequential minimal optimization)
'''
def platt_smo():
    ''' complete implementation of Platt's SMO algorithm '''
    pass

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
            fXi = float(multiply(alphas, labels_mat).T *
                        (dset_mat * dset_mat[i, :].T)) + b
            Ei = fXi - float(labels_mat[i])
            if ((labels_mat[i] * Ei < -tol) and (alphas[i] < const)) or \
               ((labels_mat[i] * Ei > tol) and (alphas[i] > const)):
                j = random_int(i, m)
                fXj = float(multiply(alphas, labels_mat).T *
                            (dset_mat * dset_mat[j, :].T)) + b
                Ej = fXj - float(labels_mat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labels_mat[i] != labels_mat[j]):
                    L = max(0, int(alphas[j] - alphas[i]))
                    H = min(const, const + int(alphas[j] - alphas[i]))
                else:
                    L = max(0, int(alphas[j] + alphas[i]) - const)
                    H = min(const, int(alphas[j] + alphas[i]))
                if L == H:
                    print('L == H')
                    continue
                eta = 2.0 * dset_mat[i, :] * dset_mat[j, :].T - \
                    dset_mat[i, :] * dset_mat[i, :].T - \
                    dset_mat[j, :] * dset_mat[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= labels_mat[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labels_mat[j] * labels_mat[i] * \
                    (alphaJold - alphas[j])
                b1 = b - Ei - labels_mat[i] * (alphas[i] - alphaIold) * \
                     dset_mat[i, :] * dset_mat[i, :].T - \
                     labels_mat[j] * (alphas[j] - alphaJold) * \
                     dset_mat[i, :] * dset_mat[j, :].T
                b2 = b - Ej - labels_mat[i] * (alphas[i] - alphaIold) * \
                     dset_mat[i, :] * dset_mat[j, :].T - \
                     labels_mat[j] * (alphas[j] - alphaJold) * \
                     dset_mat[j, :] * dset_mat[j, :].T
                if (0 < alphas[i]) and (const > alphas[i]):
                    b = 1
                elif (0 < alphas[j]) and (const > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('iter: {} i: {} pairs changed: {}'.format(
                    iter, i, alpha_pairs_changed))
        if (alpha_pairs_changed == 0):
            iter += 1
        else:
            iter = 0
        print('iteration: {}'.format(iter))

    return b, alphas


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
    dataset, labels = load_dset(filename)

    b, alphas = simple_smo(dataset, labels, 0.6, 0.001, 40)

    print(b)
    print(alphas[alphas > 0])
    print(shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataset[i], labels[i])


main('/media/gtron/files/ml/ml/classification/' +
     'support_vector_machines/testSet.txt')
