'''
    naive bayes -- classifying with conditional probability

    choosing the decision with the highest probability
'''


def load_dataset():
    '''
        Word list to vector
    '''
    posting_list = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak',
                        'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    class_vector = [0, 1, 0, 1, 0, 1]

    return posting_list, class_vector


def create_vocabulary_list(dataset):
    '''
        Create list of all unique words in all documents
    '''
    vocabulary_set = set([])

    for document in dataset:
        vocabulary_set = vocabulary_set | set(document)

    return list(vocabulary_set)


def word_set_to_vector(vocabulary_list, input_set):
    '''
        Vectorize vocabulary list
    '''
    return_vector = [0] * len(vocabulary_list)

    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] = 1
        else:
            print('The word: {} is not in my Vocabulary!'.format(word))

    return return_vector
