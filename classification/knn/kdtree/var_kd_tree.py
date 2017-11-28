from numpy import var, array, where


def kd_tree(data, chk_var=True, var_i=None, cnt=0):
    if chk_var == True:
        variances = var(data, axis=0)
        variance_sort_indices = variances.argsort()
        var_i = variance_sort_indices
        chk_var = False

    try:
        k = len(data[0])
    except IndexError as e:
        return None
    
    axis = where(var_i == (cnt % k))[0][0]

    sorted_data = data[data[:, axis].argsort()]

    median = len(data) // 2

    return {
        'location': sorted_data[median],
        'left_child': kd_tree(sorted_data[:median], chk_var, var_i, (cnt + 1)),
        'right_child': kd_tree(sorted_data[(median + 1):], chk_var, var_i, (cnt + 1))
    }




# variance search compares columns while searching tree



def knn_search_tree(tree, test):
    def distance(test_node, current_node):
        return (((test_node - current_node)**2).sum(axis=0))**0.5

    root = tree['location']
    current_best = root
    at_node = tree
    while True:
        current_best_distance = distance(test, current_best)

        if at_node['left_child'] == None and at_node['right_child'] == None:
            break
        elif at_node['left_child'] == None:
            at_node = at_node['right_child']
            if current_best_distance > distance(test, at_node['location']):
                current_best = at_node['location']
            else:
                break
        elif at_node['right_child'] == None:
            at_node = at_node['left_child']
            if current_best_distance > distance(test, at_node['location']):
                current_best = at_node['location']
            else:
                break
        else:
            left_child = at_node['left_child']['location']
            right_child = at_node['right_child']['location']
            left_dist = distance(test, left_child)
            right_dist = distance(test, right_child)

            # if the left and right distance are equal an infinite loop
            # would be created -- what to do?
            if left_dist > right_dist:
                current_best = right_child
                at_node = at_node['right_child']
            elif right_dist > left_dist:
                current_best = left_child
                at_node = at_node['left_child']
    
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

    training_set = normalized_dataset[test_dataset_size:]

    tree = kd_tree(training_set)

    count_same = 0
    for i in range(test_dataset_size):
        # find 3 nearest neighbors
        normalized_test_node = (data[:test_dataset_size][i] - min_vals) / ranges
        search_results = knn_search_tree(tree, normalized_test_node)

        from numpy import where
        kd_label = labels[label_indices[where(((training_set[:, 0] == search_results[0]) &
                                        (training_set[:, 1] == search_results[1]) &
                                        (training_set[:, 2] == search_results[2])))[0]]][0]
        print(kd_label)

        from knn import classify0

        result = classify0(normalized_test_node, training_set,
                        label_indices[test_dataset_size:], 3)

        print(labels[result])
        print(labels[label_indices[i]])

        if labels[result] == kd_label:
            count_same += 1
    print(count_same/100)