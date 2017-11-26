import numpy as np

# input can be a Python list or NumPy array
def normalize(data):
  # convert to numpy array if not of type np.ndarray
  if not isinstance(data, np.ndarray):
    data = np.array(data)

  # normalize
  min_vals = data.min(0)
  max_vals = data.max(0)
  ranges = max_vals - min_vals
  m = data.shape[0]
  data_less_min_vals = data - np.tile(min_vals, (m, 1))
  normalized_data = data_less_min_vals / np.tile(ranges, (m, 1))
  
  return normalized_data, ranges, min_vals, max_vals
