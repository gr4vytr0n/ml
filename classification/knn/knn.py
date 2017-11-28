from numpy import tile
import operator


def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = tile(inX, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat**2
    squared_distances = sq_diff_mat.sum(axis=1)
    distances = squared_distances**0.5
    sorted_distancess = distances.argsort()

    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distancess[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]
