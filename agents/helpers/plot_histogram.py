#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotHistogramRT:
    """
    Crea un histograma y lo actualiza cada que se llama update en la misma figura.
    """

    def __init__(self, series_number, bins, label_names, pause=0.0):
        self.bins = bins
        self.pause = pause
        self.series_number = series_number
        self.width = 0.7
        self.label_names = label_names
        self.this_fig_num = 20
        self.fig = plt.figure(self.this_fig_num)
        plt.show(block=False)

    def update(self, new_x, new_y):
        """
        Actualiza el histograma
        :param new_y: numpy con los nuevos valores en y [[1, 2, 3], [4, 5, 6]]
        :param new_x: numpy con los nuevos valores en x [1, 2, 3]
        :return:
        """

        new_y = np.asarray(new_y)

        # valida que las medidas nuevas sean las mismas que las establecidas en el constructor
        if new_y.shape[0] != self.series_number \
                or new_y.shape[1] != self.bins \
                or new_x.shape[0] != self.bins:
            raise ValueError('Incorrect new_y or new_x shape')

        # print np.asarray(new_x).shape
        #
        fig = plt.figure(self.this_fig_num)
        plt.clf()

        ax1 = plt.subplot(3, 1, 1)
        for ind in range(len(new_y)):
            plt.bar(new_x, new_y[ind], width=self.width, label=self.label_names[ind])
        plt.grid(True)
        plt.ylabel('c51')

        ax2 = plt.subplot(3, 1, 2)
        for ind in range(len(new_y)):
            plt.plot(new_x, np.cumsum(new_y[ind]),
                    label=self.label_names[ind])
        plt.grid(True)
        plt.ylabel('CDF')

        # prob of alive given b
        ax3 = plt.subplot(3, 1, 3)
        for ind in range(len(new_y)):
            plt.plot(-1 * new_x, 1 - np.cumsum(new_y[ind]),
                     label=self.label_names[ind])
        plt.grid(True)
        plt.ylabel('Pr(A|B)')

        plt.legend()

        fig.canvas.draw()
        plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.4)
        plt.pause(0.00001)

# plt.subplot(3, 1, 1)
# plt.plot(t, mavg_reward)
# # plt.plot(t, middle_mavg_reward)
# # plt.plot(t, bottom_mavg_reward)
# plt.title('Windyworld Environment behavior per 1000 episodes')
# plt.ylabel('Average reward')
#
# plt.subplot(3, 1, 2)
# plt.plot(t, mavg_steps)
# # plt.plot(t, middle_mavg_steps)
# # plt.plot(t, bottom_mavg_steps)
# plt.ylabel('Average steps')
#
# plt.subplot(3, 1, 3)
# plt.plot(t, end_count, label='learned')
# # plt.plot(t, middle_end_count, label='middle')
# # plt.plot(t, bottom_end_count, label='bottom')
# plt.xlabel('Episodes x 100')
# plt.ylabel('Average failure')
#
# plt.legend()
# plt.show()


if __name__ == "__main__":
    plotter = PlotHistogramRT(3, 51)
    x = np.arange(51)

    # plotter.update(x, y)
    while True:
        y = np.random.randn(3, 51)
        plotter.update(x, y)
