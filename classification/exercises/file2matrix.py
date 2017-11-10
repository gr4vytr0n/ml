import numpy as np
from sklearn import preprocessing

def file2matrix(filename):
  fr = open(filename)
  numLines = len(fr.readlines())
  matrix = np.zeros((numLines, 3))
  labels = []
  fr = open(filename)
  index = 0
  for line in fr.readlines():
    line = line.strip()
    lineList = line.split('\t')
    matrix[index, :] = lineList[0:3]
    labels.append(lineList[-1])
    index += 1
  le = preprocessing.LabelEncoder()
  le.fit(labels)
  labels = le.transform(labels)
  return matrix, labels

print(file2matrix('datingTestSet.txt'))
