'''
    using scikit-learn for k Nearest Neighbors
'''

from sklearn.neighbors import KNeighborsClassifier

def knn_classify(train, test, targets):
    '''
        classify using k Nearest Neighbor algorithm
    '''

    neigh = KNeighborsClassifier(n_neighbors=3)

    neigh.fit(train, targets)

    return neigh.predict(test)
