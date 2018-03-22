#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from agents.c51_grid_agent import C51GridAgent, run_all
from agents.safe_grid_agent import SafeGridAgent
from agents.risky_grid_agent import RiskyGridAgent


def plot_avg_std(ax, results):
    return ax.errorbar(
        range(results.shape[1]),
        np.average(results, 0),
        yerr=np.std(results, 0),
        errorevery=5
    )


class Experimenter:
    """
    Runs an experiment until Done is set to true by ai gym using an agent
    """

    def __init__(self, num_experiments=500, max_episodes=20, init_b=2):
        self.num_experiments = num_experiments
        self.max_episodes = max_episodes
        self.init_b = init_b

    def run(self, agent):

        r = np.zeros((self.num_experiments, self.max_episodes + 1))
        print(r.shape)

        # repeats experiment num_experiments times
        for exp_i in range(self.num_experiments):
            episode = 0
            b = 1
            r[exp_i, 0] = b
            #
            while not episode >= self.max_episodes:
                result = agent.run_episode()
                b += result['sum_reward']
                r[exp_i, episode + 1:] = b / self.init_b
                episode += 1
                if b <= 0:
                    break

        return r


if __name__ == "__main__":

    exp = Experimenter()

    safe_results = exp.run(agent=SafeGridAgent())
    risky_results = exp.run(agent=RiskyGridAgent())

    # print(safe_results)
    # print(risky_results)

    fig, axs = plt.subplots(nrows=2, sharex=True)

    plot_avg_std(axs[0], safe_results)
    plot_avg_std(axs[0], risky_results)
    # plot_avg_std(axs[1], results)

    plt.show()
