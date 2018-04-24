import numpy as np
import matplotlib.pyplot as plt
import sys


a = np.array([[1, 2, 2],[5,5, 1]])
print(a)
print(a.shape)
b = np.array([[4, 4, 1], [1, 1, 2]])
print(b)
print(b.shape)
c = a + b
print(c)

test = np.transpose(np.where(c == 3))
print(test)