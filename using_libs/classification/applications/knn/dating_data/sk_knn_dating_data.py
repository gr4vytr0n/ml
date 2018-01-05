'''
    use pandas to process data
    and run scikit-learn version of knn on data
'''

from os import chdir, getcwd
from sys import path

path.insert(0, getcwd() + '/using_libs/classification/knn/')

from knn import knn_classify as classify

import pandas as pd
import numpy as np

from sklearn import preprocessing


def load_dataset():
    '''
        use pandas to load csv file
    '''

    save_cwd = getcwd()
    chdir(save_cwd + '/datasets/')

    df = pd.read_csv('datingTestSet.txt', sep='\t',
                     encoding='CP1252', header=None)

    chdir(save_cwd)

    return df.iloc[:, :-1], df.iloc[:, -1]


def test():
    '''
        run test script
    '''

    dataset, labels = load_dataset()
    
    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    n_neighbor = classify(dataset, [[1000, 0.5, 340]], encoded_labels)
    
    print(le.inverse_transform(n_neighbor))
