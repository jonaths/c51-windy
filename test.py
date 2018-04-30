import numpy as np
import matplotlib.pyplot as plt
import sys


a = np.array([1,2,3,4,5,6])
b = np.array([np.where(a == k) for k in [3, 4]]).flatten()
print(b)