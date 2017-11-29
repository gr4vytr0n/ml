from numpy import array, tile, ndarray

# input can be a Python list or NumPy array


def normalize(data):
    min_vals = data.min(0)
    max_vals = data.max(0)
    ranges = max_vals - min_vals
    m = data.shape[0]
    data_less_min_vals = data - tile(min_vals, (m, 1))
    normalized_data = data_less_min_vals / tile(ranges, (m, 1))

    # need to return the mean and the variance for normalizing test data
    return normalized_data, ranges, min_vals, max_vals
