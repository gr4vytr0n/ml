import matplotlib
import matplotlib.pyplot as plt

def plot_data(data):
  figure = plt.figure()
  ax = figure.add_subplot(111)
  ax.scatter(data[:, 1], data[:, 2])
  plt.show()
