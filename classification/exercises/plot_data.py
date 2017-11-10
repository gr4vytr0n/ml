import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_data(data, labels, categories):
  figure = plt.figure()
  ax = figure.add_subplot(111)
  ax.scatter(data[:,1], data[:,0], 15.0*np.array(labels), 15.0*np.array(labels))
  plt.show()
