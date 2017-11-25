from kdtree import kdtree
from preprocess_data import preprocess_data
from normalize import normalize

matrix, labels, categories = preprocess_data('datingTestSet.txt')
normalized_matrix, ranges, min_vals, max_vals  = normalize(matrix)

tree = kdtree(normalized_matrix)

def searchTree(tree):
  return 'search tree'

print(searchTree(tree))
