#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from agents.c51_grid_agent import C51GridAgent, run_all
from agents.safe_grid_agent import SafeGridAgent
from agents.risky_grid_agent import RiskyGridAgent


# pseudocodigo
# definir b[]
# definir thres
# definir maximo numero de episodios
# definir v inicial
# para cada presupuesto

class BudgetValueIterator:
    """
    Runs an experiment until Done is set to true by ai gym using an agent
    """

    def __init__(self, max_iterations=2000, init_b=2):
        self.max_iterations = max_iterations
        self.init_b = init_b
        self.v_estimates = None
        self.budget_support = None
        self.reward_support = None
        self.reward_z_concat = None
        self.gamma = 0.95

    def retrieve_estimate_from_val(self, vals_to_search):
        """
        Receives an array of scalar and retrieve the closest values given the
        current suppoert.
        :param vals_to_search: i.e. [1, 2.5]
        :return:
        """
        vals_to_search = np.atleast_1d(vals_to_search)
        indices = np.abs(
            np.subtract.outer(self.budget_support, vals_to_search)).argmin(0)
        out = self.v_estimates[indices]
        return out, indices

    def estimate_v(self, b):
        """
        One step in value iteration process where states are the budget levels
        in the support.
        :param b: current budget
        :param current_v:
        :return:
        """
        support_plus_reward = [
            r + self.gamma * self.retrieve_estimate_from_val([r + b])[0][0]
            for r in self.reward_support]

        # array with size num_actions
        v = np.sum(
            np.multiply(self.reward_z_concat, np.array(support_plus_reward)),
            axis=1)

        return np.max(v)

    def update_v(self, sample_b, val):
        indices = self.retrieve_estimate_from_val([sample_b])
        # position [1] is indices, index 0 because one val passed
        self.v_estimates[indices[1][0]] = val

    def run(self, agent_name):

        if agent_name == 'c51':
            agent = C51GridAgent()
        else:
            sys.exit(0)

        # retrieve learned model predictions
        agent.reset()
        self.reward_z_concat, q, _, _ = agent.agent.predict(agent.s_t)
        self.reward_support = agent.agent.z

        # budget support
        self.budget_support = range(0, 21, 1)

        # assume the initial values are zero
        self.v_estimates = np.zeros((len(self.budget_support)))

        print("support")
        print(self.budget_support)

        iteration = 0
        b = self.init_b
        v_min_thres = 0.1

        # repeats experiment num_experiments times
        budgets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        for b in budgets:
            iteration = 0
            while not iteration >= self.max_iterations:
                print(str(b), "===")
                print(iteration)

                estimate = self.estimate_v(b)
                current_estimate = np.copy(self.v_estimates)

                self.update_v(b, estimate)

                diff = current_estimate - self.v_estimates
                print("estimate")
                print(estimate)
                print("val")
                print(self.v_estimates)
                print("diff")
                print(diff)

                b = b + estimate
                iteration += 1
                if b <= 0 or b > 40:
                    b = self.init_b


        plt.plot(self.budget_support, self.v_estimates)
        plt.show()

        #
        #     # aqui voy... hacer funcion dummy para recuperar valores.
        #     # calcular v desde v inicial cuando se termine el presupuesto
        #     # establecer threshold y terminar cuando se cumpla o cuando num_experiments
        #
        # new_v_estimate = self.estimate_v(b, self.v_estimate)

        # v_delta = abs(new_v_estimate - self.v_estimate)
        # self.v_estimate = new_v_estimate
        #
        # if v_delta < v_min_thres:
        #     break
        #
        # print(r)
        # return r, None
        return 1, 2


if __name__ == "__main__":

    # budgets = [0, 2, 4, 8]
    budgets = [0]

    for b in range(len(budgets)):
        print(
            str(budgets[b]) + " ==============================================")

        exp = BudgetValueIterator(init_b=budgets[b])
        support, values = exp.run(agent_name='c51')
