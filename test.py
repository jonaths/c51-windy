import numpy as np
import matplotlib.pyplot as plt
import sys


a = np.array(
    [
        [
            [1., 2., 2., 1.],
            [1., 5., 1., 7.],
        ],
        [
            [4., 4., 1., 8.],
            [1., 1., 2., 1.],
        ],
        [
            [4., 2., 1., 8.],
            [1., 1., 3., 5.],
        ]
    ]
)
print(a)

argmax = np.argmax(a, axis=1).astype(float)
print(argmax)
print(argmax.shape)
mask = np.squeeze(np.diff(a, axis=1) == 0, axis=1)
print(mask)

argmax[mask] = 0.5
print(argmax)