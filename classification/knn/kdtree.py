import numpy

def kdtree(data, leafsize=10):
  ndim = data.shape[0]
  ndata = data.shape[1]

  # hyper-rectangle or is it hyperplane ??
  hrect = numpy.zeros((2, ndim))
  hrect[0,:] = data.min(axis=1)
  hrect[1,:] = data.max(axis=1)

  # root
  idx = numpy.argsort(data[0,:], kind='mergesort')
  print(numpy.sort(data, axis=-1))
  data[:,:] = data[:,idx]
  splitval = data[ndim//2,:]

  # left_hrect = hrect.copy()
  # right_hrect = hrect.copy()
  # left_hrect[1,0] = splitval


if __name__ == '__main__':
  from preprocess_data import preprocess_data
  from normalize import normalize

  matrix, labels, categories = preprocess_data('datingTestSet.txt')
  normalized_matrix, ranges, min_vals, max_vals  = normalize(matrix)
  
  kdtree(normalized_matrix)
