# need to learn more about tensorflow before I can do kNN with it

import numpy as np
import tensorflow as tf

dataset = np.array([[0, 0.2], [1.0, 0], [0, 1.0], [1.0, 0.5]])
testset = np.array([0, 0.5])

k = 3

dsetSize = dataset.shape[0]  
distance = np.square(np.add(np.tile(testset, (4, 1)), np.negative(dataset))).sum(axis=1).argsort()

prediction = tf.argmin(tf.convert_to_tensor(distance, np.int64), 0)
print(prediction)
sess = tf.Session()
sess.run(prediction)


