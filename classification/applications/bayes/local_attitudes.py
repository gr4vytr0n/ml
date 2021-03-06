'''
    Using naive bayes to reveal local attitudes from
    personal ads
'''
from random import uniform
from operator import itemgetter
from numpy import array
from feedparser import parse

from os import getcwd, chdir
from sys import path
path.insert(0, getcwd() + '/classification/bayes/')
from bayes import tokenize, create_vocabulary_list, word_to_vector, trainer, classifier


def load_stop_words():
    ''' load stop words list '''
    save_cwd = getcwd()
    chdir(save_cwd + '/classification/applications/bayes')

    with open('english_stop_words.txt') as stop_words:
        stop_words_list = [l.strip().split('\n') for l in stop_words]

    chdir(save_cwd)

    return stop_words_list


def calculate_frequency(vocabulary_list, full_text):
    '''
        calculates how many times each word occurs in text
        returns top 30 words
    '''
    occurences = {}
    for token in vocabulary_list:
        occurences[token] = full_text.count(token)

    return sorted(occurences.items(), key=itemgetter(1), reverse=True)[:30]


def process_feeds(feeds):
    ''' extract and tokenize feed data '''
    feed_0 = feeds[0]['entries']
    feed_1 = feeds[1]['entries']

    min_length = min(len(feed_0), len(feed_1))

    classes = []
    processed_feeds = []

    for i in range(min_length):
        processed_feeds.append(tokenize(feed_0[i]['summary']))
        classes.append(0)
        processed_feeds.append(tokenize(feed_1[i]['summary']))
        classes.append(1)

    return processed_feeds, classes


def local_attitudes(in_feeds, in_classes):
    '''
        access two RSS feeds
        to find local frequency words used in personal ads
    '''
    feeds = in_feeds[:]
    classes = in_classes[:]

    vocabulary_list = create_vocabulary_list(feeds)

    top_30_words = calculate_frequency(vocabulary_list, feeds)

    # remove most frequently occuring words
    for most_freq in top_30_words:
        if most_freq[0] in vocabulary_list:
            vocabulary_list.remove(most_freq[0])

    # remove stop words
    stop_words = load_stop_words()
    for stop_word in stop_words:
        if stop_word in vocabulary_list:
            vocabulary_list.remove(stop_word)

    training_set = []
    training_classes = []

    test_set = []
    test_classes = []

    for _ in range(20):
        rand_idx = int(uniform(0, len(feeds)))

        test_set.append(feeds[rand_idx])
        test_classes.append(classes[rand_idx])

        del feeds[rand_idx]
        del classes[rand_idx]

    training_classes = classes

    for feed in feeds:
        training_set.append(word_to_vector(
            vocabulary_list, feed, word_occurences='bag'))

    p0v, p1v, p_jargon = trainer(array(training_set), array(training_classes))

    error_count = 0

    for idx, feed in enumerate(test_set):
        word_vector = word_to_vector(
            vocabulary_list, feed, word_occurences='bag')

        if classifier(array(word_vector), p0v, p1v, p_jargon) != test_classes[idx]:
            error_count += 1

    return vocabulary_list, p0v, p1v, float(error_count) / len(test_set)


def get_top_words(feeds, classes):
    '''
        display locally used words
    '''
    vocabulary_list, p0v, p1v, p_ = local_attitudes(feeds, classes)
    
    top_ny = []
    top_sf = []

    for i in range(len(p0v)):
        if p0v[i] > -6.0:
            top_ny.append((vocabulary_list[i], p0v[i]))

        if p1v[i] > -6.0:
            top_sf.append((vocabulary_list[i], p1v[i]))

    print('NY*'*10)
    for item in sorted(top_ny, key=lambda pair: pair[1], reverse=True):
        print(item[0])

    print('SF*'*10)
    for item in sorted(top_sf, key=lambda pair: pair[1], reverse=True):
        print(item[0])


def test():
    ''' run script '''
    feeds = []

    feeds.append(parse('http://newyork.craigslist.org/stp/index.rss'))
    feeds.append(parse('http://sfbay.craigslist.org/stp/index.rss'))

    processed_feeds, classes = process_feeds(feeds)

    get_top_words(processed_feeds, classes)

    err_result = 0.0
    for _ in range(10):
        vl_, p0_, p1_, err = local_attitudes(processed_feeds, classes)

        err_result += err

    print(err_result / 10)
