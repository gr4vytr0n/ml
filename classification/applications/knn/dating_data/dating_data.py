from numpy import array
from sys import path
from os import getcwd

path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn/utils')
path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn')
from preprocess_data import preprocess_data
from knn import classify0


data, normalizing, labeling = preprocess_data(
    getcwd() + '/dating_data', 'datingTestSet.txt')
n_array, ranges, min_vals, max_vals = normalizing
label_indices, labels = labeling


# Classify a person with classifier


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    gaming = float(input('percent time playing video games?'))
    flyerMiles = float(input('frequent flyer miles earned each year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    inArr = array([flyerMiles, gaming, iceCream])
    classifierResult = classify0(
        (inArr - min_vals) / ranges, n_array, label_indices, 3)
    print('You will probably like this person: ',
          resultList[classifierResult - 1])


classifyPerson()
