import numpy as np

# adapted from wikipedia
class Node(object):
  def __init__(self, location, left_child, right_child):
    self.location = location
    self.left_child = left_child
    self.right_child = right_child

  def __str__(self):
    return 'LOCATION: {} LEFT: {} RIGHT: {}'.format(self.location, self.left_child, self.right_child)

def kdtree(data, depth = 0):
  try:
    k = len(data[0])
  except IndexError as e:
    return None
  # axis of feature to split on
  axis = depth % k 

  # sort data on axis
  sortedData = data[data[:, axis].argsort()]

  # median for split point
  median = len(data) // 2

  # create node and construct subtrees
  return Node(
    location = sortedData[median],
    left_child = kdtree(sortedData[:median], (depth + 1)),
    right_child = kdtree(sortedData[(median + 1):], (depth + 1))
  )

if __name__ == '__main__':
  from preprocess_data import preprocess_data
  from normalize import normalize

  matrix, labels, categories = preprocess_data('datingTestSet.txt')
  normalized_matrix, ranges, min_vals, max_vals  = normalize(matrix)
  
  print(kdtree(normalized_matrix))
  #print(kdtree(np.array([(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)])))
