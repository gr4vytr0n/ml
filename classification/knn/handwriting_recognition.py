from os import listdir, chdir, getcwd
from numpy import zeros, array

from knn import classify0

def read_files(directory, filenames):
  saveCwd = getcwd()
  chdir(directory)

  files = []
  for file in filenames:
    with open(file) as f:
      files.append(f.read())

  chdir(saveCwd)
  return files

def get_filenames(directories):
  filenames = []
  for directory in directories:
    filenames.append(listdir(directory))
  return filenames

def prep_data(data):
  vectors = zeros((len(data), 1024))
  for h in range(len(data)):
    lines = data[h].split('\n')
    vector = zeros((1, 1024))
    for i in range(32):
      line = lines[i]
      for j in range(32):
        vector[0, 32*i+j] = int(line[j])
    vectors[h, :] = vector
  return vectors

def classifiyHandwriting(testData, trainingData, filenames):
  testLabels = []
  trainingLabels = []
  for filename in filenames[0]:
    label = int(filename.split('.')[0].split('_')[0])
    testLabels.append(label)
  for filename in filenames[1]:
    label = int(filename.split('.')[0].split('_')[0])
    trainingLabels.append(label)
  errorCount = 0.0
  for test in range(len(testData)):
    classifierResult = classify0(testData[test], trainingData, trainingLabels, 3)
    print('classifier result: {}, real answer: {}'.format(classifierResult, testLabels[test]))
    if (classifierResult != testLabels[test]):
      errorCount += 1.0
  print('total errors: {}'.format(errorCount))
  print('total error rate: {}'.format(errorCount/float(len(testData))))


if __name__ == '__main__':
  directories = ['testDigits', 'trainingDigits']
  filenames = get_filenames(directories)
  testData = prep_data(read_files(directories[0], filenames[0]))
  trainingData = prep_data(read_files(directories[1], filenames[1]))
  classifiyHandwriting(testData, trainingData, filenames)
