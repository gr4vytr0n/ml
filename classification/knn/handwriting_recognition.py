from os import listdir, chdir, getcwd

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

if __name__ == '__main__':
  directories = ['testDigits', 'trainingDigits']
  filenames = get_filenames(directories)
  testData = read_files(directories[0], filenames[0])
  trainingData = read_files(directories[1], filenames[1])
