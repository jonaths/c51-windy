import numpy as np
import matplotlib.pyplot as plt
import sys


a = np.array([1,2])
a = np.stack(([a] * 4), axis=1)
print(a)