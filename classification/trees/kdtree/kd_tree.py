from numpy import array

def kd_tree(data, depth = 0):
    try:
        k = len(data[0])
    except IndexError as e:
        return None

    # axis of feature to split on
    axis = depth % k

    # sort data on axis
    sorted_data = data[data[:, axis].argsort()]

    # median for split point
    median = len(data) // 2

    # create node and construct subtrees
    return {
        'location': sorted_data[median],
        'left_child': kd_tree(sorted_data[:median], (depth + 1)),
        'right_child': kd_tree(sorted_data[(median + 1):], (depth + 1))
    }

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

            if left_dist >= right_dist:
                current_best = right_child
                at_node = at_node['right_child']
            elif right_dist > left_dist:
                current_best = left_child
                at_node = at_node['left_child']
    
    return current_best
