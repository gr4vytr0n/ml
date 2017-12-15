'''
    hand written digits classification using kNN
'''


from sys import path
from os import getcwd, listdir, chdir
from time import perf_counter
from numpy import zeros, array

path.insert(0, '/media/gtron/files/ml/ml/classification/knn/')
from knn import classify


def build_file_lists(path, dirs):
    ''' insert filenames from given directories into a list '''
    save_cwd = getcwd()
    chdir(path)

    # test at index 0, training at index 1
    filenames = [listdir(dir) for dir in dirs]

    chdir(path+dirs[0])
    test_files = []
    for number_file in filenames[0]:
        with open(number_file) as n_file:
            test_files.append(n_file.read())

    chdir(path+dirs[1])
    train_files = []    
    for number_file in filenames[1]:
        with open(number_file) as n_file:
            train_files.append(n_file.read())

    chdir(save_cwd)

    return test_files, train_files, filenames


def to_vectors(data):
    vectors = zeros((len(data), 1024))
    for h in range(len(data)):
        lines = data[h].split('\n')
        vector = zeros((1, 1024))
        for i in range(32):
            line = lines[i]
            for j in range(32):
                vector[0, 32*i+j] = int(line[j])
        vectors[h, :] = vector
    return vectors


def classify_handwriting(testData, trainingData, filenames):
    testLabels = []
    trainingLabels = []
    for filename in filenames[0]:
        label = int(filename.split('.')[0].split('_')[0])
        testLabels.append(label)
    for filename in filenames[1]:
        label = int(filename.split('.')[0].split('_')[0])
        trainingLabels.append(label)
    errorCount = 0.0
    for test in range(len(testData)):
        classifierResult = classify(testData[test], trainingData, trainingLabels, 3)
        # print('classifier result: {}, real answer: {}'.format(classifierResult, testLabels[test]))
        if (classifierResult != testLabels[test]):
            errorCount += 1.0
    print('total errors: {}'.format(errorCount))
    print('total error rate: {}'.format(errorCount/float(len(testData))))


# create arrays of sample vectors for test and training data
file_path = '/media/gtron/files/ml/ml/datasets/hw/'
file_dirs = ['testDigits', 'trainingDigits']
test_files, train_files, filenames = build_file_lists(file_path, file_dirs)
test_set = to_vectors(test_files)
train_set = to_vectors(train_files)

t0 = perf_counter()

classify_handwriting(test_set, train_set, filenames)

t1 = perf_counter()
print('elapsed time: {}'.format(t1 - t0))
