import numpy as np
from operator import itemgetter

from normalize import normalize


data = np.random.rand(24).reshape((8,3))

normData, ranges, min_vals, max_vals = normalize(data)

print(normData.sort(key=itemgetter(axis=0)))
