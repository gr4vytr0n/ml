'''
    decision tree building algorithm
'''

from math import log
import operator


def calc_entropy(dataset):
    '''
        measurement of information
    '''
    num_entries = len(dataset)

    # dictionary of frequency of class occurences
    label_counts = {}
    for sample in dataset:
        curr_label = sample[-1]
        if curr_label not in label_counts.keys():
            label_counts[curr_label] = 0
        label_counts[curr_label] += 1

    # use label_counts to calculate probability of that class
    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries

        # calculate entropy and sum this up for all classes
        entropy -= prob * log(prob, 2)

    return entropy


def split(dataset, axis, value):
    '''
        split tree on a feature
        ID3 algorithm
    '''
    result = []

    for sample in dataset:
        if sample[axis] == value:
            # remove feature from new list
            reduced_sample = sample[:axis]
            reduced_sample.extend(sample[axis + 1:])

            result.append(reduced_sample)

    return result


def best_split(dataset):
    '''
        determine best feature to split dataset on
    '''
    num_features = len(dataset[0]) - 1

    # calculate entropy before splitting dataset
    # base disorder
    base_entropy = calc_entropy(dataset)

    # split on feature and get entropy
    # then compare to base entropy
    # do this for each feature
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # feature values from each sample
        feature_values = [example[i] for example in dataset]

        # get unique values for feature
        uniq_values = set(feature_values)

        # split on feature and calculate entropy
        feature_entropy = 0.0
        for value in uniq_values:
            sub_dataset = split(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            feature_entropy += prob * calc_entropy(sub_dataset)

        # measure information gain (reduction in entropy/messiness)
        info_gain = base_entropy - feature_entropy

        # compare current info gain to base and choose best
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_count(class_list):
    '''
        determine class by majority count
    '''
    class_count = {}

    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def create_tree(dataset, orig_labels):
    '''
        recursively create decision tree
    '''
    # create copy of labels
    labels = orig_labels.copy()

    # list of sample classes
    class_list = [example[-1] for example in dataset]

    # stop if all classes are equal
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # return majority when no more features
    if len(dataset[0]) == 1:
        return majority_count(class_list)

    # determine best feature to split on
    best_feature = best_split(dataset)
    best_feature_label = labels[best_feature]

    # recursively build tree
    tree = {best_feature_label: {}}

    # remove feature from those available to split on
    del(labels[best_feature])

    # split dataset on best_feature and call create_tree with subset
    feature_values = [example[best_feature] for example in dataset]
    uniq_values = set(feature_values)
    for value in uniq_values:
        sub_labels = labels[:]
        tree[best_feature_label][value] = create_tree(
            split(dataset, best_feature, value), sub_labels)
    
    return tree
