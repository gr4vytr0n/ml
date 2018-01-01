'''
    use decision tree to classifiy
'''
from os import getcwd
from  sys import path
path.insert(0, getcwd() + '/classification/trees/')
from decision_trees import create_tree

def create_dataset():
    ''' create dataset '''
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    
    labels = ['no surfacing', 'flippers']

    return dataset, labels


def classify(input_tree, feature_labels, test_vector):
    '''
        classifier
    '''

    current_feature = list(input_tree.keys())[0]
    branches = input_tree[current_feature]

    # index of feature from labels list
    feature_index = feature_labels.index(current_feature)

    # traverse tree
    for key in branches.keys():
        if test_vector[feature_index] == key:
            if type(branches[key]).__name__ == 'dict':
                # continue traversing tree to find class
                class_label = classify(branches[key], feature_labels,
                                       test_vector)
            else:
                # found class at leaf node
                class_label = branches[key]
    
    return class_label

def test():
    '''
        run script
    '''
    dataset, labels = create_dataset()

    tree = create_tree(dataset, labels)

    print(classify(tree, labels, [1, 1]))
