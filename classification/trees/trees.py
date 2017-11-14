from math import log

def calcShannonEntropy(dataset):
  numEntries = len(dataset)
  labelCounts = {}
  for featVec in dataset:
    currentLabel = featVec[-1]
    if currentLabel not in labelCounts.keys():
      labelCounts[currentLabel] = 0
    labelCounts[currentLabel] += 1
  shannonEnt = 0.0
  for key in labelCounts:
    prob = float(labelCounts[key]) / numEntries
    shannonEnt -= prob * log(prob, 2)
  return shannonEnt

def splitDataset(dataset, axis, value):
  retDataset = []
  for featVec in dataset:
    if featVec[axis] == value:
      reducedFeatVec = featVec[:axis]
      reducedFeatVec.extend(featVec[axis+1:])
      retDataset.append(reducedFeatVec)
  return retDataset

def chooseBestFeatureToSplit(dataset):
  numFeatures = len(dataset[0]) - 1
  baseEntropy = calcShannonEntropy(dataset)
  bestInfoGain = 0.0
  bestFeature = -1
  for i in range(numFeatures):
    featList = [example[i] for example in dataset]
    uniqueVals = set(featList)
    newEntropy = 0.0
    for value in uniqueVals:
      subDataset = splitDataset(dataset, i, value)
      prob = len(subDataset) / float(len(dataset))
      newEntropy += prob * calcShannonEntropy(subDataset)
    infoGain = baseEntropy - newEntropy
    if (infoGain > bestInfoGain):
      bestInfoGain = infoGain
      bestFeature = i
  return bestFeature

if __name__ == '__main__':
  def createDataset():
    dataset = [ [1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no'] ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels

  data, labels = createDataset()

  print(chooseBestFeatureToSplit(data))