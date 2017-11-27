import numpy as np

from kdtree import kdtree

# only classifies one test sample at a time


def searchTree(tree, test):
    def distance(testNode, currentNode):
        return (((testNode - currentNode)**2).sum(axis=0))**0.58

    # build array of nodes from tree
    # first node is root
    # then choose next node from left and right based on nearest distance (add to array of nodes)
    # continue until None is reached on left and right
    # do knn classify0 on array finding kNN

    result = []

    # traverse tree
    # add root node to result list
    root = tree['location']
    result.append(root)

    atNode = tree
    while True:
        if atNode['left_child'] == None and atNode['right_child'] == None:
            break
        elif atNode['left_child'] == None:
            atNode = atNode['right_child']
            result.append(atNode['location'])
        elif atNode['right_child'] == None:
            atNode = atNode['left_child']
            result.append(atNode['location'])
        else:
            # measure distance from test node to left and right children
            # add closest distance to result list
            left_child = atNode['left_child']['location']
            right_child = atNode['right_child']['location']
            left_dist = distance(test, left_child)
            right_dist = distance(test, right_child)

            if left_dist > right_dist:
                # right is closer
                result.append(right_child)
                atNode = atNode['right_child']
            elif right_dist > left_dist:
                #left is closer
                result.append(left_child)
                atNode = atNode['left_child']

    return np.array(result)


if __name__ == '__main__':
    from sys import path
    path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn/utils')
    path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn')
    from knn import classify0
    from preprocess_data import preprocess_data

    data, normalizing, labeling = preprocess_data('dating_data', 'datingTestSet.txt')
    normalized_matrix, ranges, min_vals, max_vals = normalizing
    label_indices, labels = labeling

    tree = kdtree(normalized_matrix[1:])
    
    testNode = data[0]
    nTestData = (testNode - min_vals) / ranges

    searchResults = searchTree(tree, nTestData)
    # new search method
    print(labels[label_indices[np.where(((normalized_matrix[:,0]==searchResults[0][0]) &
        (normalized_matrix[:,1]==searchResults[0][1]) &
        (normalized_matrix[:,2]==searchResults[0][2])))[0]]])
    searchResultsLabels = []
    for i in searchResults.tolist():
        # these index searches are probably really slowing this down
        searchResultsLabels.append(label_indices[1: normalized_matrix.tolist()[1:].index(i)])


    classIndex = classify0(nTestData, searchResults, searchResultsLabels, 3)
    print(labels[classIndex])
    # classLabel = labels[classIndex]
    # print(nTestData, '<--', resultList[classLabel - 1],
    #       '   ', matrix[classIndex].tolist(), '<-- Nearest')

    # data = np.array(list(map(list, [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)])))
    # nData, rData, minData, maxData =  normalize(data)
    # trainingDataLabels =['BAD', 'GOOD', 'GOOD', 'BAD', 'GOOD', 'GOOD']
    # tree = kdtree(nData)
    # nTestData = (np.array([1,3])-minData)/rData
    # searchResults = searchTree(tree, nTestData)
    # searchResultsLabels = []
    # for i in searchResults.tolist():
    #   searchResultsLabels.append(nData.tolist().index(i))
    # classIndex = classify0(nTestData, searchResults, searchResultsLabels, 1)
    # classLabel = trainingDataLabels[classIndex]
    # print('[1, 3] <--', classLabel,'   ', data[classIndex], '<-- Nearest')
