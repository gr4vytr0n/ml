'''
    use decision tree to classifiy what type
    of contact lenses a person should use
'''

from os import getcwd, chdir
from  sys import path
path.insert(0, getcwd() + '/classification/trees')
from decision_trees import *

def process_dataset():
    ''' prepare dataset for building tree '''
    save_cwd = getcwd()

    chdir(save_cwd + '/datasets/')
    with open('lenses.txt') as file:
        dataset = [i.strip().split('\t') for i in file.readlines()]
    
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']

    chdir(save_cwd)

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
        
    dataset, labels = process_dataset()

    tree = create_tree(dataset, labels)

    print(classify(tree, labels, ['young', 'hyper', 'no', 'normal']))
