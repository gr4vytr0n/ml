from numpy import var, array, where

def variances(data):
    variances = var(data, axis=0)

    return variances.argsort()

def kd_tree(data, var_i, cnt=0):
    try:
        k = len(data[0])
    except IndexError as e:
        return None
    
    axis = where(var_i == (cnt % k))[0][0]

    sorted_data = data[data[:, axis].argsort()]

    median = len(data) // 2

    return {
        'location': sorted_data[median],
        'bounds': [sorted_data[0], sorted_data[-1]],
        'left_child': kd_tree(sorted_data[:median], var_i, (cnt + 1)),
        'right_child': kd_tree(sorted_data[(median + 1):], var_i, (cnt + 1))
    }

def knn_search_tree(tree, t_node, variance_sort_indices, k):
    def test_value(t_node, current_node):
        return t_node < current_node

    def test_bounds(bounds):
        print(bounds)

    root = tree['location']
    current_best = root
    at_node = tree

    cnt = 0
    while True:
        axis = where(variance_sort_indices == (cnt % k))[0][0]
        current_best_test = test_value(t_node[axis], current_best[axis])
        print(at_node['bounds'])
        if at_node['left_child'] == None and at_node['right_child'] == None:
            break
        elif at_node['left_child'] == None:
            at_node = at_node['right_child']
            if test_value(t_node[axis], at_node['location'][axis]):
                current_best = at_node['location']
            else:
                break
        elif at_node['right_child'] == None:
            at_node = at_node['left_child']
            if test_value(t_node[axis], at_node['location'][axis]):
                current_best = at_node['location']
            else:
                break
        else:
            left_child = at_node['left_child']['location']
            right_child = at_node['right_child']['location']
            left_dist = t_node[axis] - left_child[axis]
            right_dist = t_node[axis] - right_child[axis]

            # if the left and right distance are equal an infinite loop
            # would be created -- what to do?
            if left_dist >= right_dist:
                current_best = right_child
                at_node = at_node['right_child']
            elif right_dist > left_dist:
                current_best = left_child
                at_node = at_node['left_child']
    cnt += 1

    return current_best


if __name__ == '__main__':
    from sys import path
    path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn/utils')
    path.insert(0, '/media/gtron/files/ml/mlclone/classification/knn')

    from preprocess_data import preprocess_data

    data, normalizing, labeling = preprocess_data(
        'dating_data', 'datingTestSet.txt')
    normalized_dataset, ranges, min_vals, max_vals = normalizing
    label_indices, labels = labeling

    # set size of test set
    ho_ratio = 0.1
    m = data.shape[0]
    test_dataset_size = int(m * ho_ratio)
    training_set= normalized_dataset[test_dataset_size:]
    variance = variances(training_set)
    tree = kd_tree(training_set, variance)

    count_same = 0
    for i in range(test_dataset_size):
        # find 3 nearest neighbors
        normalized_test_node = (data[:test_dataset_size][i] - min_vals) / ranges
        search_results = knn_search_tree(tree, normalized_test_node, variance, len(normalized_test_node))
        
        from numpy import where
        kd_label = labels[label_indices[where(((training_set[:, 0] == search_results[0]) &
                                        (training_set[:, 1] == search_results[1]) &
                                        (training_set[:, 2] == search_results[2])))[0]]][0]
        print(search_results)
        from knn import classify0

        result = classify0(normalized_test_node, training_set,
                        label_indices[test_dataset_size:], 3)

        # print(labels[result])
        # print(labels[label_indices[i]])

        if labels[result] == kd_label:
            count_same += 1
    print(count_same/100)