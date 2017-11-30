'''
    kd_tree: build a k-D tree
    nn_search_tree: search for nearest neigbor in k-D tree
'''

from numpy import array, where


def kd_tree(data, cnt=0, variance=[]):
    '''
        Build a k-D tree
    '''
    try:
        k = len(data[0])
    except IndexError:
        return None

    if variance == []:
        # axis of feature to split on
        axis = cnt % k
    else:
        axis = where(variance == (cnt % k))[0][0]

    # sort data on axis
    sorted_data = data[data[:, axis].argsort()]

    # median for split point
    median = len(data) // 2

    # create node and construct subtrees
    return {
        'location': sorted_data[median],
        'left_child': kd_tree(sorted_data[:median], (cnt + 1), variance),
        'right_child': kd_tree(sorted_data[(median + 1):], (cnt + 1), variance)
    }


def nn_search_tree(tree, t_node, variance=[]):
    '''
        Find nearest neighbor in k-D tree
    '''
    root = tree['location']
    at_node = tree
    current_best = root

    if variance == []:
        def distance(test_node, current_node):
            return (((test_node - current_node)**2).sum(axis=0))**0.5

    else:
        def test_value(test_node, current_node):
            return test_node < current_node

    cnt = 0
    while True:
        if variance == []:
            current_best_distance = distance(t_node, current_best)
        else:
            k = len(t_node)
            axis = where(variance == (cnt % k))[0][0]
            current_best_test = test_value(t_node[axis], current_best[axis])

        if at_node['left_child'] == None and at_node['right_child'] == None:
            break
        elif at_node['left_child'] == None:
            at_node = at_node['right_child']
            if variance == []:
                if current_best_distance > distance(t_node, at_node['location']):
                    current_best = at_node['location']
                else:
                    break
            else:
                if test_value(t_node[axis], at_node['location'][axis]):
                    current_best = at_node['location']
                else:
                    break
        elif at_node['right_child'] == None:
            at_node = at_node['left_child']
            if variance == []:
                if current_best_distance > distance(t_node, at_node['location']):
                    current_best = at_node['location']
                else:
                    break
            else:
                if test_value(t_node[axis], at_node['location'][axis]):
                    current_best = at_node['location']
                else:
                    break
        else:
            left_child = at_node['left_child']['location']
            right_child = at_node['right_child']['location']

            if variance == []:
                left_dist = distance(t_node, left_child)
                right_dist = distance(t_node, right_child)

                if left_dist >= right_dist:
                    current_best = right_child
                    at_node = at_node['right_child']
                elif right_dist > left_dist:
                    current_best = left_child
                    at_node = at_node['left_child']
            else:
                left_test = abs(t_node[axis] - left_child[axis])
                right_test = abs(t_node[axis] - right_child[axis])

                if left_test >= right_test:
                    current_best = right_child
                    at_node = at_node['right_child']
                elif right_test > left_test:
                    current_best = left_child
                    at_node = at_node['left_child']
        cnt += 1

    return current_best
