import numpy as np

from kdtree import kdtree
from knn import classify0
from preprocess_data import preprocess_data
from normalize import normalize

labelCategories = ['not at all', 'in small doses', 'in large doses']

matrix, labels, categories = preprocess_data('datingTestSet.txt')
normalized_matrix, ranges, min_vals, max_vals  = normalize(matrix)

tree = kdtree(normalized_matrix)
practiceTree = kdtree(np.array([(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]))

searchParameters = [34.0, 4400.0, 0.3]

def searchTree(tree, parameters):
  return tree

print(searchTree(practiceTree, searchParameters))
print(labelCategories[classify0(searchParameters, normalized_matrix, labels, 3)])