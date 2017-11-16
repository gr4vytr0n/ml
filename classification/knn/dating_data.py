from preprocess_data import preprocess_data
from normalize import normalize
from plot_data import plot_data
from to_matrix import to_matrix
from knn import classify0
from numpy import array

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

#test()

# Classify a person with classifier
def classifyPerson():
  resultList = ['not at all', 'in small doses', 'in large doses']
  gaming = float(input('percent time playing video games?'))
  flyerMiles = float(input('frequent flyer miles earned each year?'))
  iceCream = float(input('liters of ice cream consumed per year?'))
  inArr = array([flyerMiles, gaming, iceCream])
  classifierResult = classify0((inArr-min_vals)/ranges, normalized_matrix, labels, 3)
  print('You will probably like this person: ', resultList[classifierResult - 1])

classifyPerson()