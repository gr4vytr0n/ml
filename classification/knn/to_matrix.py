import numpy as np

def to_matrix(filename):
  fr = open(filename)
  num_lines = len(fr.readlines())
  matrix = np.zeros((num_lines, 3))

  fr = open(filename)
  index = 0
  for line in fr.readlines():
    line = line.strip()
    line_list = line.split('\t')
    matrix[index, :] = line_list[0:3]
    index += 1

  return matrix
