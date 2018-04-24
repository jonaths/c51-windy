#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
import random

from agents.c51_grid_agent import C51GridAgent, run_all
from agents.safe_grid_agent import SafeGridAgent
from agents.risky_grid_agent import RiskyGridAgent

import matplotlib.pylab as pylab
params = {
    'legend.fontsize': 'xx-large',
    'figure.figsize': (15, 5),
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large'
    }
pylab.rcParams.update(params)


def plot_avg_std(ax, results, title, linestyle='-', errorevery=5, label=None):
    ax.set_title(title)
    return ax.errorbar(
        range(results.shape[1]),
        np.average(results, 0),
        yerr=np.std(results, 0),
        errorevery=errorevery,
        linestyle=linestyle,
        label=label
    )

def plot_reward(ax, results, title, linestyle='-', errorevery=5, label=None):
    results = np.insert(np.diff(results, axis=1), 0, results[:, 0], axis=1)
    ax.set_title(title)
    return ax.errorbar(
        range(results.shape[1]),
        np.average(results, 0),
        yerr=np.std(results, 0),
        errorevery=errorevery,
        linestyle=linestyle,
        label=label
    )

def plot_trajectories(ax, results, title, label=None):
    ax.set_title(title)
    return ax.plot(
        range(results.shape[1]),
        np.transpose(results[np.random.choice(len(results), size=1, replace=False)]),
        label=label
    )

def plot_failure_rate(ax, results, title, linestyle='-'):
    failures = (results[:, -1] <= 0) * 1
    ax.set_title(title)
    return ax.plot(
        range(results.shape[0]), np.cumsum(failures), linestyle=linestyle)


class Experimenter:
    """
    Runs an experiment until Done is set to true by ai gym using an agent
    """

    def __init__(self, num_experiments=100, max_episodes=10, init_b=2):
        self.num_experiments = num_experiments
        self.max_episodes = max_episodes
        self.init_b = init_b


if __name__ == "__main__":

    # agent = C51GridAgent()
    # while True:
    #     agent.run_episode(4)

    b = 0
    fig_avg, axs_avg = plt.subplots(nrows=2, sharex=True)

    print(str(b)+" ==============================================")

    exp = Experimenter(init_b=b)

    safe_results = np.load('results/safe_' + str(b) + '.npy')
    c51_results = np.load('results/c51_' + str(b) + '.npy')
    risky_results = np.load('results/risky_' + str(b) + '.npy')

    print(risky_results.shape)

    # index = np.transpose(np.where(risky_results < 0))[random.randint(0, 98)][0]
    index = 33
    print(index)

    safe_results = np.array([safe_results[index]])
    c51_results = np.array([c51_results[index]])
    risky_results = np.array([risky_results[index]])

    ax = plot_reward(axs_avg[0], safe_results, 'b0=' + str(b),
                     '-', label="safe", errorevery=5)
    ax = plot_reward(axs_avg[0], c51_results, 'b0=' + str(b),
                     '-.', label="mod c51")
    ax = plot_reward(axs_avg[0], risky_results, 'b0=' + str(b),
                     ':', label="risky", errorevery=4)

    ax = plot_avg_std(axs_avg[1], safe_results, 'b0='+str(b), '-', label="safe", errorevery=5)
    ax = plot_avg_std(axs_avg[1], c51_results, 'b0=' + str(b), '-.', label="budget c51")
    ax = plot_avg_std(axs_avg[1], risky_results, 'b0='+str(b), ':', label="risky", errorevery=4)

    plt.legend()

    # fig_avg.suptitle('Average final budget per episode')
    fig_avg.tight_layout()

    plt.show()
