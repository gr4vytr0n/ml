import numpy

def kdtree(data, leafsize=10):
  ndim = data.shape[0]
  ndata = data.shape[1]
  print(ndim, ndata)

  # find bounding hyper-rectangle
  hrect = numpy.zeros((2, ndim))
  print(hrect)
  
  hrect[0,:] = data.min(axis=1)
  hrect[1,:] = data.max(axis=1)

  print(hrect)

if __name__ == '__main__':
  from preprocess_data import preprocess_data
  matrix, labels, categories = preprocess_data('datingTestSet.txt')

  kdtree(matrix)
