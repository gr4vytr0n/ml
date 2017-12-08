'''
    Using naive bayes to reveal local attitudes from
    personal ads
'''
from random import uniform
from operator import itemgetter
from numpy import array
from feedparser import parse
from bayes import tokenize, create_vocabulary_list, word_to_vector, trainer, classifier

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
    # --------  try not doin this ------------
    # for most_freq in top_30_words:
    #     if most_freq[0] in vocabulary_list:
    #         vocabulary_list.remove(most_freq[0])

    training_set = []
    training_classes = []

    test_set = []
    test_classes = []

    for _ in range(20):
        rand_idx = int(uniform(0, len(feeds)))
        print(len(feeds), rand_idx)
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

def main():
    ''' run script '''
    feeds = []

    feeds.append(parse('http://newyork.craigslist.org/stp/index.rss'))
    feeds.append(parse('http://sfbay.craigslist.org/stp/index.rss'))
    
    processed_feeds, classes = process_feeds(feeds)

    err_result = 0.0
    for _ in range(10):
        vl_, p0_, p1_, err = local_attitudes(processed_feeds, classes)
        print(err)
        err_result += err

    print(err_result / 10)

main()
