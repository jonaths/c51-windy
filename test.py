import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_xy(ax, data, color):

    # data = np.arange(0, 10).reshape(2, 5)

    X = data[:,:-1]
    Y = data[:,1:]
    # print(data.shape)
    #
    # data = data.reshape(data.shape[0] * data.shape[1], 1)
    # print(data.shape)
    #
    # data = np.append(data, np.roll(data, 1)).reshape(2, data.shape[0] * data.shape[1])
    # print(data)
    #
    # print data.shape
    return ax.scatter(X, Y, color=color, s=5)
    # return ax.scatter(data[0], data[1], color=color, s=5)


plt.grid(True)
fig, ax = plt.subplots(nrows=1, sharex=True)

# plot_xy(ax[0], np.load('results/c51_0.npy'), color='red')
# plot_xy(ax[0], np.load('results/c51_1.npy'), color='blue')
# plot_xy(ax[0], np.load('results/c51_2.npy'), color='green')
# plot_xy(ax[0], np.load('results/c51_3.npy'), color='black')


# plot_xy(ax[1], np.load('results/risky_0.npy'), color='red')
# plot_xy(ax[1], np.load('results/risky_1.npy'), color='blue')
# plot_xy(ax[1], np.load('results/risky_2.npy'), color='green')
# plot_xy(ax[1], np.load('results/risky_3.npy'), color='black')

# plot_xy(ax[2], np.load('results/safe_0.npy'), color='red')
# plot_xy(ax[2], np.load('results/safe_1.npy'), color='blue')
# plot_xy(ax[2], np.load('results/safe_2.npy'), color='green')
# plot_xy(ax[2], np.load('results/safe_3.npy'), color='black')

plot_xy(ax, np.load('results/c51_0.npy'), color='red')
plot_xy(ax, np.load('results/risky_0.npy'), color='blue')
plot_xy(ax, np.load('results/safe_0.npy'), color='green')

plt.show()
