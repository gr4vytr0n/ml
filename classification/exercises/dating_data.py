from preprocess_data import preprocess_data
from plot_data import plot_data

matrix, label_categories, categories = preprocess_data('datingTestSet.txt')

plot_data(matrix)
