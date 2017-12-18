'''
    Spam classifier using naive bayes classification
'''

from random import uniform
from numpy import zeros, ones, array, log
from os import getcwd
from sys import path
path.insert(0, getcwd() + '/classification/bayes/')

from bayes import tokenize, create_vocabulary_list, \
    word_to_vector, trainer, classifier


def load_document(filename):
    '''
        Open file(s) and return list of contents
    '''
    read_file = ''
    with open(filename, encoding='Windows-1252') as document:
        read_file = document.read()

    return read_file


def process_documents():
    '''
        process documents
        returns ham and spam documents as tuple
    '''
    abs_path = getcwd() + '/datasets/'
    ham_documents = []
    spam_documents = []
    for i in range(1, 26):
        ham_documents.append(tokenize(load_document(
            '{}email/ham/{}.txt'.format(abs_path, str(i)))))
        spam_documents.append(tokenize(load_document(
            '{}email/spam/{}.txt'.format(abs_path, str(i)))))

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

        del docs_copy[rand_idx]
        del classes[rand_idx]

    training_classes = classes

    for doc in docs_copy:
        training_set.append(word_to_vector(
            vocabulary_list, doc, word_occurences='set'))

    p0v, p1v, p_spam = trainer(array(training_set), array(training_classes))

    error_count = 0

    for idx, doc in enumerate(test_set):
        word_vector = word_to_vector(
            vocabulary_list, doc, word_occurences='set')
        if classifier(array(word_vector), p0v, p1v, p_spam) != test_classes[idx]:
            error_count += 1

    return float(error_count) / len(test_set)


def main():
    ''' run script '''
    hams, spams = process_documents()

    err_result = 0.0
    for _ in range(10):
        err_result += spam_test(hams + spams)

    print(err_result / 10)


main()
