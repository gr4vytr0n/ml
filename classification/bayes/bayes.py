'''
    naive bayes -- classifying with conditional probability

    choosing the decision with the highest probability
'''
from re import split
from numpy import zeros, ones, array, log


def load_document(filename):
    '''
        Open file(s) and return list of contents
    '''
    read_file = ''
    with open(filename, encoding='Windows-1252') as document:
        read_file = document.read()

    return read_file


def tokenize(the_string):
    '''
        Tokenize string
    '''
    tokens_list = split(r'\W*', the_string)

    return [tokens.lower() for tokens in tokens_list if len(tokens) > 2]


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


def word_bag_to_vector(vocabulary_list, input_set):
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
        Returns: the probability of class 0, the probability of class 1,
                 and probability of being abusive
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
    prob_1 = sum(test_vector * class_1_vector) + log(class_1_prob)
    prob_0 = sum(test_vector * class_0_vector) + log(1.0 - class_1_prob)

    if prob_1 > prob_0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    from random import uniform

    def process_documents():
        ''' 
            process documents
            returns ham and spam documents as tuple
        '''
        abs_path = '/media/gtron/files/ml/ml/classification/bayes/'
        ham_documents = []
        spam_documents = []
        for i in range(1, 26):
            ham_documents.append(tokenize(load_document('{}email/ham/{}.txt'.format(abs_path, str(i)))))
            spam_documents.append(tokenize(load_document('{}email/spam/{}.txt'.format(abs_path, str(i)))))
    
        return ham_documents, spam_documents
    

    
    def spam_test(documents):
        ''' test documents for spam '''
        vocabulary_list = create_vocabulary_list(documents)

        classes = zeros(25, dtype=int).tolist() + ones(25, dtype=int).tolist()

        docs_copy = documents[:]

        training_set = []
        training_classes = []

        test_set = []
        test_classes = []

        for _ in range(10):
            rand_idx = int(uniform(0, len(docs_copy)))
            test_set.append(docs_copy[rand_idx])
            test_classes.append(classes[rand_idx])
            del(docs_copy[rand_idx])
            del(classes[rand_idx])
        
        training_classes = classes

        for doc in docs_copy:
            training_set.append(word_set_to_vector(vocabulary_list, doc))
        
        p0v, p1v, p_spam = trainer(array(training_set), array(training_classes))

        error_count = 0

        for idx, doc in enumerate(test_set):
            word_vector = word_set_to_vector(vocabulary_list, doc)
            if classifier(array(word_vector), p0v, p1v, p_spam) != test_classes[idx]:
                error_count += 1
        
        return float(error_count) / len(test_set)


    hams, spams = process_documents()

    err_result = 0.0
    for _ in range(10):
        err_result += spam_test(hams + spams)

    print(err_result / 10)

