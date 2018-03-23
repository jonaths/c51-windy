#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from agents.c51_grid_agent import C51GridAgent, run_all
from agents.safe_grid_agent import SafeGridAgent
from agents.risky_grid_agent import RiskyGridAgent


def plot_avg_std(ax, results, title):
    ax.set_title(title)
    return ax.errorbar(
        range(results.shape[1]),
        np.average(results, 0),
        yerr=np.std(results, 0),
        errorevery=5
    )


def plot_failure_rate(ax, results, title):
    failures = (results[:, -1] <= 0) * 1
    ax.set_title(title)
    return ax.plot(
        range(results.shape[0]), np.cumsum(failures))

class Experimenter:
    """
    Runs an experiment until Done is set to true by ai gym using an agent
    """

    def __init__(self, num_experiments=100, max_episodes=10, init_b=2):
        self.num_experiments = num_experiments
        self.max_episodes = max_episodes
        self.init_b = init_b

    def run(self, agent):

        # array of num_experiments x max_episodes + 1 which contains the
        # final budget for each experiment
        r = np.zeros((self.num_experiments, self.max_episodes + 1))
        # a plain vector that contains the cell id where each episode ended
        end_count = []

        # repeats experiment num_experiments times
        for exp_i in range(self.num_experiments):
            episode = 0
            b = self.init_b
            r[exp_i, 0] = b
            # repeats for a given amount of episodes
            while not episode >= self.max_episodes:
                result = agent.run_episode()
                # adds current reward to total experiment budget
                b += result['sum_reward']
                # adds the last position to the end_count list
                end_count.append(result['step_seq'][-1])
                # all remaining elements in list are set to the last accumulated
                # budget
                r[exp_i, episode + 1:] = b
                episode += 1
                # ends if it ran out of budget in this experiment
                if b <= 0:
                    break

        return r, end_count


if __name__ == "__main__":

    budgets = [0, 2, 4]
    _, axs_avg = plt.subplots(nrows=len(budgets))
    _, axs_failure = plt.subplots(nrows=len(budgets))

    for b in range(len(budgets)):

        exp = Experimenter(init_b=budgets[b])

        safe_results, _ = exp.run(agent=SafeGridAgent())
        c51_results, _ = exp.run(agent=C51GridAgent())
        risky_results, _ = exp.run(agent=RiskyGridAgent())

        print("safe")
        print(safe_results)
        print("risky")
        print(risky_results)

        plot_avg_std(axs_avg[b], safe_results, 'b='+str(budgets[b]))
        plot_avg_std(axs_avg[b], c51_results, 'b=' + str(budgets[b]))
        plot_avg_std(axs_avg[b], risky_results, 'b='+str(budgets[b]))

        plot_failure_rate(axs_failure[b], safe_results, 'b='+str(budgets[b]))
        plot_failure_rate(axs_failure[b], c51_results, 'b=' + str(budgets[b]))
        plot_failure_rate(axs_failure[b], risky_results, 'b='+str(budgets[b]))

    plt.show()
