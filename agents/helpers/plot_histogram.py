#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams.update({'font.size': 16})


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
        if new_y.shape[1] != self.series_number \
                or new_y.shape[2] != self.bins \
                or new_x.shape[0] != self.bins:
            raise ValueError('Incorrect new_y or new_x shape')

        # print np.asarray(new_x).shape
        #
        # fig = plt.figure(self.this_fig_num)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
        fig = plt.figure(self.this_fig_num)

        plt.clf()

        # ax1 = plt.subplot(4, 1, 1)
        for ind in range(len(new_y[0])):
            ax1.bar(new_x, new_y[0][ind], width=self.width, label=self.label_names[ind])
        plt.grid(True)
        ax1.set_ylabel('c51')
        plt.legend()

        # ax2 = plt.subplot(4, 1, 2)
        for ind in range(len(new_y[0])):
            ax2.plot(new_x, np.cumsum(new_y[0][ind]),
                     label=self.label_names[ind])
        plt.grid(True)
        ax2.set_ylabel('CDF')
        plt.legend()

        # prob of alive given b
        # ax3 = plt.subplot(4, 1, 3)
        for ind in range(len(new_y[0])):
            ax3.plot(+1 * new_x, new_y[2][ind],
                     label=self.label_names[ind])
        plt.grid(True)
        ax3.set_ylabel('Beta')
        plt.legend()

        # ax3_a = plt.subplot(4, 1, 4, sharex=ax1)
        # ax3_b = ax3_a.twinx()
        for ind in range(len(new_y[0])):
            # modified q values
            # ax3_b.plot(-1 * new_x, new_y[3][ind], linestyle='--')
            # ax3_b.set_ylabel('Q')
            # q values
            ax4.plot(+1 * new_x, new_y[1][ind])
            ax4.set_ylabel('Qopt')
        plt.legend()
        plt.grid(True)

        # top = 0.973,
        # bottom = 0.068,
        # left = 0.122,
        # right = 0.978,
        # hspace = 0.595,
        # wspace = 0.2

        fig.canvas.draw()
        plt.tight_layout()
        plt.pause(0.00001)


if __name__ == "__main__":
    plotter = PlotHistogramRT(3, 51)
    x = np.arange(51)

    # plotter.update(x, y)
    while True:
        y = np.random.randn(3, 51)
        plotter.update(x, y)
