from preprocess_data import preprocess_data
from normalize import normalize
from plot_data import plot_data
from knn import classify0

matrix, labels, categories = preprocess_data('datingTestSet.txt')

normalized_matrix, ranges, min_vals, max_vals  = normalize(matrix)

#plot_data(normalized_matrix, label_categories, categories)

def test():
  ho_ratio = 0.10 # 10% of data for test data
  m = normalized_matrix.shape[0]
  num_test_vectors = int(m * ho_ratio)
  error_count = 0.0
  for i in range(num_test_vectors):
    classifier_results = classify0(normalized_matrix[i, :], normalized_matrix[num_test_vectors:m, :], \
      labels[num_test_vectors:m], 3)
    print('classifier answer: %d, real answer: %d' % (classifier_results, labels[i]))
    if (classifier_results != labels[i]):
      error_count += 1.0
  print('total error rate: %f' % (error_count / float(num_test_vectors)))    

test()