import numpy as np

def get_labels(filename):
  labels = []
  
  fr = open(filename)
  index = 0
  for line in fr.readlines():
    line = line.strip()
    line_list = line.split('\t')
    labels.append(line_list[-1])
    index += 1

  return labels
