from to_matrix import to_matrix
from get_labels import get_labels
from categorize_labels import categorize_labels

def preprocess_data(filename):
  matrix = to_matrix(filename)
  labels = get_labels(filename)
  label_categories, categories = categorize_labels(labels)

  return matrix, label_categories, categories
