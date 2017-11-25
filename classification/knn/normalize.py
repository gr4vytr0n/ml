import numpy as np

def normalize(data):
  min_vals = data.min(0)
  max_vals = data.max(0)
  ranges = max_vals - min_vals
  normalized_data = np.zeros(np.shape(data))
  m = data.shape[0]
  normalized_data = data - np.tile(min_vals, (m, 1))
  normalized_data = normalized_data / np.tile(ranges, (m, 1))
  # ranges, min_vals, max_vals not normalized when returned
  return normalized_data, ranges, min_vals, max_vals
