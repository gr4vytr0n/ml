'''
    Insert paths from project
'''

from sys import path


def insert_paths():
    '''
        Does the inserting
    '''
    path.insert(0, '/media/gtron/files/ml/mlclone/datasets')
    path.insert(0, '/media/gtron/files/ml/mlclone/utils')
    path.insert(0, '/media/gtron/files/ml/mlclone/classification')
    path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn')
    path.insert(0, '/media/gtron/files/ml/mlclone/classification/trees')
