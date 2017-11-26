import numpy as np

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
  return {
    'location': sortedData[median],
    'left_child': kdtree(sortedData[:median], (depth + 1)),
    'right_child': kdtree(sortedData[(median + 1):], (depth + 1))
  }
