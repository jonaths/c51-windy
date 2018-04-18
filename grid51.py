#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from agents.c51_grid_agent import C51GridAgent, run_all
from agents.safe_grid_agent import SafeGridAgent
from agents.risky_grid_agent import RiskyGridAgent


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

def plot_trajectories(ax, results, title, label=None):
    ax.set_title(title)
    return ax.plot(
        range(results.shape[1]),
        np.transpose(results[np.random.choice(len(results), size=50, replace=False)]),
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

    def __init__(self, num_experiments=100, max_episodes=100, init_b=2):
        self.num_experiments = num_experiments
        self.max_episodes = max_episodes
        self.init_b = init_b

    def run(self, agent_name):

        if agent_name == 'c51':
            agent = C51GridAgent()
        elif agent_name == 'risky':
            agent = RiskyGridAgent()
        elif agent_name == 'safe':
            agent = SafeGridAgent()
        else:
            sys.exit(0)

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

                agent.reset()

                # runs episode starting with budget b
                result = agent.run_episode(b)
                num_steps = len(result['step_seq'])
                print("Result: ",
                      "exp_i:", str(exp_i),
                      "episode:", str(episode),
                      "steps:", str(num_steps)
                )
                print(result)
                if num_steps > 4:
                    sys.exit(0)
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
                    print("Game done (Budget) --------------------------------")
                    # sys.exit(0)
                    break
            print("Game done (Episodes) --------------------------------------")
        return r, end_count


if __name__ == "__main__":

    # agent = C51GridAgent()
    # while True:
    #     agent.run_episode(4)

    budgets = [0, 2, 4, 8]
    fig_avg, axs_avg = plt.subplots(nrows=len(budgets), sharex=True, sharey=True)
    fig_failure, axs_failure = plt.subplots(nrows=len(budgets), sharex=True, sharey=True)

    for b in range(len(budgets)):

        print(str(budgets[b])+" ==============================================")

        exp = Experimenter(init_b=budgets[b])

        safe_results, _ = exp.run(agent_name='safe')
        c51_results, _ = exp.run(agent_name='c51')
        risky_results, _ = exp.run(agent_name='risky')

        # safe_results = np.load('results/safe_' + str(b) + '.npy')
        # c51_results = np.load('results/c51_' + str(b) + '.npy')
        # risky_results = np.load('results/risky_' + str(b) + '.npy')

        # print("safe")
        # print(safe_results)
        # print("c51")
        # print(c51_results)
        # print("risky")
        # print(risky_results)

        # np.save('results/safe_'+str(b)+'.npy', safe_results)
        # np.save('results/c51_' + str(b) + '.npy', c51_results)
        # np.save('results/risky_' + str(b) + '.npy', risky_results)

        # ax = plot_avg_std(axs_avg[b], safe_results, 'b0='+str(budgets[b]), '-', label="safe", errorevery=5)
        # ax = plot_avg_std(axs_avg[b], c51_results, 'b0=' + str(budgets[b]), '-.', label="r")
        # ax = plot_avg_std(axs_avg[b], risky_results, 'b0='+str(budgets[b]), ':', label="risky", errorevery=4)

        ax = plot_trajectories(axs_avg[b], c51_results, 'b0=' + str(budgets[b]), label="r")


        plt.legend()

        ax = plot_failure_rate(axs_failure[b], safe_results, 'b0='+str(budgets[b]), '-')
        ax = plot_failure_rate(axs_failure[b], c51_results, 'b0=' + str(budgets[b]), '-.')
        ax = plot_failure_rate(axs_failure[b], risky_results, 'b0='+str(budgets[b]), ':')

        plt.legend()

    fig_avg.suptitle('Average final budget per episode')
    fig_failure.suptitle('Failure count per experiment')

    fig_avg.tight_layout()
    fig_failure.tight_layout()

    plt.show()
