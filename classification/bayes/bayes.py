'''
    naive bayes -- classifying with conditional probability

    choosing the decision with the highest probability
'''

from numpy import ones, log, array


def load_dataset():
    '''
        Word vectors from lists
    '''
    documents = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak',
                  'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classes = [0, 1, 0, 1, 0, 1]

    return documents, classes


def create_vocabulary_list(dataset):
    '''
        Create lists of unique words from all documents
    '''
    vocabulary_set = set([])

    for document in dataset:
        vocabulary_set = vocabulary_set | set(document)

    return list(vocabulary_set)


def word_set_to_vector(vocabulary_list, input_set):
    '''
        Vectorize vocabulary list -- set of words
    '''
    return_vector = [0] * len(vocabulary_list)

    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] = 1
        else:
            print('The word: {} is not in my Vocabulary!'.format(word))

    return return_vector

def word_bag_to_vector(vocabulary_list,input_set):
    '''
        Vectorize vocabulary list -- bag of words
    '''
    return_vector = [0] * len(vocabulary_list)

    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] += 1
    
    return return_vector


def trainer(train_array, train_category):
    '''
        Calculating probabilities from vectors
        Returns: the probability of class 0, the probability of class 1, and probability of being abusive

    '''
    number_words = len(train_array[0])

    class_0_num = ones(number_words)
    class_1_num = ones(number_words)
    class_0_denom = 2.0
    class_1_denom = 2.0

    for idx, value in enumerate(train_array):
        if train_category[idx] == 1:
            class_1_num += value
            class_1_denom += sum(value)
        else:
            class_0_num += value
            class_0_denom += sum(value)
    class_1_prob = log(class_1_num / class_1_denom)
    class_0_prob = log(class_0_num / class_0_denom)

    return class_0_prob, class_1_prob, sum(train_category) / float(len(train_array))

def classifier(test_vector, class_0_vector, class_1_vector, class_1_prob):
    '''
        Naive Bayes classifier
    '''
    p1 = sum(test_vector * class_1_vector) + log(class_1_prob)
    p0 = sum(test_vector * class_0_vector) + log(1.0 - class_1_prob)

    if p1 > p0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    def test():
        ''' test module '''
        posts, classes = load_dataset()
        vocabulary_list = create_vocabulary_list(posts)

        train_array = []
        for post_in_docs in posts:
            train_array.append(word_set_to_vector(
                vocabulary_list, post_in_docs))

        p0v, p1v, abusive = trainer(array(train_array), array(classes))

        test_vector = ['love', 'my', 'dalmation']
        test_array = array(word_set_to_vector(vocabulary_list, test_vector))

        print('classification: {}'.format(classifier(test_array, p0v, p1v, abusive)))

    test()
