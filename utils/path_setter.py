'''
    Insert paths from project
'''

from sys import path


def insert_paths():
    '''
        Does the inserting
    '''
    path.insert(0, '/media/gtron/files/ml/ml/datasets')
    path.insert(0, '/media/gtron/files/ml/ml/utils')
    path.insert(0, '/media/gtron/files/ml/ml/classification')
    path.insert(0, '/media/gtron/files/ml/ml/classification/knn')
    path.insert(0, '/media/gtron/files/ml/ml/classification/trees')
