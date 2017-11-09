from numpy import *
import operator

def classify0(inX, dataset, labels, k):
    datasetSize = dataset.shape[0]

    diffMat = tile(inX , (datasetSize, 1)) - dataset
    sqDiffMat = diffMat**2
    squareDistances = sqDiffMat.sum(axis=1)
    distances = squareDistances**0.5
    sortedDistances = distances.argsort()

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistances[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]
