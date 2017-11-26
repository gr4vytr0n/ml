from sklearn import neighbors

def build_tree(data):
  tree = neighbors.KDTree(data, leaf_size=2)
  return tree

if __name__ == '__main__':
  from preprocess_data import preprocess_data
  from normalize import normalize
  from numpy import array

  matrix, labels, categories = preprocess_data('datingTestSet.txt')
  normalized_matrix, ranges, min_vals, max_vals  = normalize(matrix)
  
  labelsList = ['not at all', 'in small doses', 'in large doses']
  tree = build_tree(normalized_matrix)
  test = array([34.0, 4400.0, 0.3])
  dist, ind = tree.query([test], k=3)

  ind = ind[0]
  for i in ind:
    print(labelsList[labels[i]])