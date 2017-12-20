'''
    support vector machines
'''
from random import uniform
from numpy import mat, shape, multiply, zeros, abs, nonzero, exp, sign


'''
    SMO (sequential minimal optimization)
'''
class OptStruct:
    ''' data structure to hold properties of platt_smo '''
    def __init__(self, dataset, labels, const, tol, k_tuple):
        self.dataset = dataset
        self.labels = labels
        self.const = const
        self.tol = tol
        self.m = shape(dataset)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.e_cache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.dataset, self.dataset[i, :], k_tuple)


def calc_Ek(oS, k):
    fXk = float(multiply(oS.alphas, oS.labels).T * \
                oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labels[k])

    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    max_delta_E = 0
    Ej = 0
    oS.e_cache[i] = [1, Ei]
    valid_E_cache_list = nonzero(oS.e_cache[:, 0].A)[0]
    if (len(valid_E_cache_list)) > 1:
        for k in valid_E_cache_list:
            if k == i:
                continue
            Ek = calc_Ek(oS, k)
            delta_E = abs(Ei - Ek)
            if (delta_E > max_delta_E):
                maxK = k
                max_delta_E = delta_E
                Ej = Ek
        return maxK, Ej
    else:
        j = random_int(i, oS.m)
        Ej = calc_Ek(oS, j)
    
    return j, Ej

def updateEk(oS, k):
    Ek = calc_Ek(oS, k)
    oS.e_cache[k] = [1, Ek]

def inner_L(i, oS):
    Ei = calc_Ek(oS, i)
    if ((oS.labels[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.const)) or \
       ((oS.labels[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labels[i] != oS.labels[j]):
            L = max(0, int(oS.alphas[j] - oS.alphas[i]))
            H = min(oS.const, oS.const + int(oS.alphas[j] - oS.alphas[i]))
        else:
            L = max(0, int(oS.alphas[j] + oS.alphas[i]) - oS.const)
            H = min(oS.const, int(oS.alphas[j] + oS.alphas[i]))
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labels[j] * (Ei - Ej) / eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labels[j] * oS.labels[i] * \
                        (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labels[i] * (oS.alphas[i] - alphaIold) * \
             oS.K[i, i]  - oS.labels[j] * \
             (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labels[i] * (oS.alphas[i] - alphaIold) * \
             oS.K[i, j]  - oS.labels[j] * \
             (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.const > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.const > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def platt_smo(dataset, labels, const, tol, max_iter, k_tuple = ('lin', 0)):
    ''' complete implementation of Platt's SMO algorithm '''
    oS = OptStruct(mat(dataset), mat(labels).transpose(), const, tol, k_tuple)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(oS.m):
                alpha_pairs_changed += inner_L(i, oS)
            print('fullset, iter: {} i: {} pairs changed: {}'.format( \
                  iter, i, alpha_pairs_changed))
            iter += 1
        else:
            non_bound_Is = nonzero((oS.alphas.A > 0) * (oS.alphas.A < const))[0]
            for i in non_bound_Is:
                alpha_pairs_changed += inner_L(i, oS)
                print('non-bound, iter: {} i: {}, pairs changed: {}'.format( \
                      iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif (alpha_pairs_changed == 0):
            entire_set = True
        print('iteration number: {}'.format(iter))
    
    return oS.b, oS.alphas


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


def kernel_trans(X, A, k_tuple):
    ''' support for kernels '''
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if k_tuple[0] == 'lin':
        K = X * A.T
    elif k_tuple[0] == 'rbf':
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = exp(K / (-1 * k_tuple[1] ** 2))
    else:
        raise NameError('o shit!!! -- what kernel is this???')
    
    return K


def calc_Ws(alphas, dataset, labels):
    ''' calculate ws '''
    dst = mat(dataset)
    lbls = mat(labels).transpose()
    m, n = shape(dst)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * lbls[i], dst[i, :].T)
    return w
