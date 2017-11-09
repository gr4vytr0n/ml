from numpy import *
import operator

# k-Nearest Neighbor algorithm
# inX: value to be classified, dataset: known data, labels: class labels, k: number of nearest neighbors
def classify0(inX, dataset, labels, k):
    # number of rows
    datasetSize = dataset.shape[0]
    
    # determine distance between inX and each row in dataset using Euclidian distance formula
    # subtract inX X from datasets' rows' X and same with Ys
    diffMat = tile(inX , (datasetSize, 1)) - dataset
    # square diff of Xs and square diff of Ys
    sqDiffMat = diffMat**2
    # add X and Y of each row
    squareDistances = sqDiffMat.sum(axis=1)
    # get square root of each element of sqDiffMat array
    distances = squareDistances**0.5
    
    # get indices of distances in array of distances in increasing order
    sortedDistances = distances.argsort()
    
    # create dictionary of first k lowest distances from inX
    classCount = {}
    for i in range(k):
        # find label at index i
        voteLabel = labels[sortedDistances[i]]
        # add a key for each class and count number of times that class occurs in first k of sortedDistances
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    
    # sort classCount in decreasing order based on count for each class
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    # return majority class
    return sortedClassCount[0][0]
