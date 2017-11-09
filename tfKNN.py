import operator
import numpy as np
import tensorflow as tf

dataset = np.array([[0, 0.2], [1.0, 0], [0, 1.0], [1.0, 0.5], [1.0, 0.2], [1.0, 1.0], [0, 0], [1.0, 0.7]])
labels = ['A', 'B', 'A', 'B', 'B', 'B', 'A', 'B']

testset = np.array([1.0, 0.5])

k = 3

datasetSize = dataset.shape[0]
tiledTestset = np.tile(testset, (datasetSize, 1))
diffTrainAndTest = tf.subtract(tiledTestset, dataset)
squared = tf.square(diffTrainAndTest)
reducedSum = tf.reduce_sum(squared, axis=1)

argSorted = tf.py_func(np.argsort, [reducedSum], tf.int64, stateful=False)
argMin = tf.argmin(argSorted, 0)

sess = tf.Session()

results = sess.run(argSorted)

classCount = {}
for i in range(k):
    voteLabel = labels[results[i]]
    classCount[voteLabel] = classCount.get(voteLabel,0) + 1
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

print(sortedClassCount[0][0]) 