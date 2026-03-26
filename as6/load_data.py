import numpy as np

def load_data(file_path):
  with open(file_path,'r') as file:
    data = np.array([line.strip() for line in file.readlines()])
  return data

