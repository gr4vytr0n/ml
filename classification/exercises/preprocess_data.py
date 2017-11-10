from to_matrix import to_matrix
from get_labels import get_labels
from categorize_labels import categorize_labels


matrix = to_matrix('datingTestSet.txt')
labels = get_labels('datingTestSet.txt')
label_categories, categories = categorize_labels(labels)
