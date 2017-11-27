#from plot_data import plot_data
from numpy import array
from sys import path
from os import getcwd

path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn/utils')
path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn')
from preprocess_data import preprocess_data
from knn import classify0


n_array, labels, label_indices = preprocess_data(
    getcwd() + '/dating_data', 'datingTestSet.txt')


#plot_data(normalized_matrix, label_categories, categories)


def test():
    ho_ratio = 0.10  # 10% of data for tesing
    m = n_array.shape[0]
    num_test_vectors = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vectors):
        classifier_results = classify0(n_array[i, :], n_array[num_test_vectors:m, :],
                                       label_indices[num_test_vectors:], 3)
        print('classifier answer: {}, real answer: {}'.format(labels[classifier_results], labels[label_indices[i]]))
        if (labels[classifier_results] != labels[label_indices[i]]):
            error_count += 1.0
    print('total error rate: {}'.format(error_count / float(num_test_vectors)))


test()

# Classify a person with classifier


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    gaming = float(input('percent time playing video games?'))
    flyerMiles = float(input('frequent flyer miles earned each year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    inArr = array([flyerMiles, gaming, iceCream])
    classifierResult = classify0(
        (inArr - min_vals) / ranges, normalized_matrix, labels, 3)
    print('You will probably like this person: ',
          resultList[classifierResult - 1])


# classifyPerson()
