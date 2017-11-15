from to_matrix import to_matrix
from get_labels import get_labels
from categorize_labels import categorize_labels

def preprocess_data(filename):
  matrix = to_matrix(filename)
  labels_raw = get_labels(filename)
  labels, categories = categorize_labels(labels_raw)

  return matrix, labels, categories
