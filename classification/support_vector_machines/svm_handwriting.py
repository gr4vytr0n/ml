'''
    classify handwritten digits using svm
'''
from os import listdir, chdir, getcwd
from numpy import zeros, array, mat, nonzero, multiply, sign, shape
from support_vector_machines import platt_smo, kernel_trans


def read_files(directory, filenames):
    saveCwd = getcwd()
    chdir(directory)
    files = []
    for file in filenames:
        with open(file) as f:
            files.append(f.read())

    chdir(saveCwd)

    return files


def prep_data(data):
    vectors = zeros((len(data), 1024))
    for h in range(len(data)):
        lines = data[h].split('\n')
        vector = zeros((1, 1024))
        for i in range(32):
            line = lines[i]
        for j in range(32):
            vector[0, 32 * i + j] = int(line[j])
        vectors[h, :] = vector
    return vectors


def load_images(dir_name):
    ''' load images '''
    hw_labels = []
    training_file_list = listdir(dir_name)
    training_mat = prep_data(read_files(dir_name, listdir(dir_name)))
    for i in range(len(training_file_list)):
        filename_str = training_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)

    return training_mat, hw_labels


def test(k_tuple=('rbf', 10)):
    CWD = getcwd()
    train_dir_name = '/media/gtron/files/ml/ml/datasets/hw/trainingDigits/'
    
    dset, labels = load_images(train_dir_name)
    
    b, alphas = platt_smo(dset, labels, 200, 0.0001, 10000, k_tuple)
    d_mat = mat(dset)
    l_mat = mat(labels).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    s_vs = d_mat[sv_ind]
    l_sv = l_mat[sv_ind]
    print('there are {} support vectors'.format(shape(s_vs)[0]))
    m, n = shape(d_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, d_mat[i, :], k_tuple)
        predict = kernel_eval.T * multiply(l_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(labels[i]):
            error_count += 1
    print('the training error rate is: {}'.format(float(error_count) / m))
    test_dir_name = '/media/gtron/files/ml/ml/datasets/hw/testDigits/'
    dset, labels = load_images(test_dir_name)
    error_count = 0
    d_mat = mat(dset)
    l_mat = mat(labels).transpose()
    m, n = shape(d_mat)
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, d_mat[i, :], k_tuple)
        predict = kernel_eval.T * multiply(l_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(labels[i]):
            error_count += 1
    print('the test error rate is: {}'.format(float(error_count) / m))


test()
