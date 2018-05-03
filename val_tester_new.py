#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from agents.c51_grid_agent import C51GridAgent, run_all
# from agents.safe_grid_agent import SafeGridAgent
# from agents.risky_grid_agent import RiskyGridAgent


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

    def __init__(self, max_iterations=500):
        self.max_iterations = max_iterations
        self.v_estimates = None
        self.budget_support = None
        self.reward_support = None
        self.reward_z_concat = None
        self.diff = None
        self.gamma = 0.95

    def run(self, agent_name):

        if agent_name == 'c51':
            agent = C51GridAgent()
        else:
            sys.exit(0)

        # retrieve learned model predictions
        # agent.reset()
        self.reward_z_concat, q, _, _ = agent.agent.predict(agent.s_t)
        self.reward_support = np.array(agent.agent.z)

        # para probar
        # self.reward_z_concat = np.array([
        #     [0.1, 0.1, 0.1, 0.20, 0.5],
        #     [0.1, 0.2, 0.2, 0.5, 0.0]
        # ])
        # self.reward_support = np.array(
        #     [-2, -1, 0, 1, 2]
        # )

        # en -1 acumula lo que no esta dentro del soporte
        # por eso el for empieza desde el segundo indice
        self.budget_support = np.arange(-1, 30, 1)
        self.v_estimates = np.zeros((self.reward_z_concat.shape[0], self.budget_support.shape[0]))
        self.diff = np.zeros(self.budget_support.shape[0])
        self.diff.fill(1000.)

        print("Initial info ============================================================")
        print("budget_support")
        print(self.budget_support)
        print("reward z concat")
        print(self.reward_z_concat.shape)
        print("reward support")
        print(self.reward_support.shape)
        print("v estimates")
        print(self.v_estimates)

        input("Continue... ")

        # print(self.v_estimates[[0, 3]])
        # sys.exit(0)
        counter = 1
        while np.max(self.diff) > 0.05:
            print("================================================")
            print(counter)
            v_old = np.copy(self.v_estimates)
            # empieza desde el segundo indice para acumular lo que no esta dentro del
            # soporte
            for i in range(1, len(self.budget_support)):
                print("--------------------------------------------")
                print(i, self.budget_support[i], self.reward_support)

                # lo asigna al entero inmediato inferior
                b_next = np.floor(self.reward_support + self.budget_support[i])

                print("b_next")
                # si es menor que el minimo del soporte lo asigna al minimo del soporte
                b_next[b_next < self.budget_support[0]] = self.budget_support[0]
                # si es mayor que el maximo del soporte lo asigna al maximo del soporte
                b_next[b_next > self.budget_support[-1]] = self.budget_support[-1]
                print(b_next)

                # busca en budget_support los indices que contienen a b_next
                print("next_indices")
                next_indices = np.array([np.where(self.budget_support == k)
                                         for k in b_next]).flatten()
                print(next_indices)

                print("v_next")
                v_next = self.v_estimates[:, next_indices]
                print(v_next)

                print("inner")
                inner = self.reward_support + self.gamma * v_next
                print(inner)

                print("multiply and sum")
                print(np.multiply(self.reward_z_concat, inner))
                print(np.sum(np.multiply(self.reward_z_concat, inner), axis=1))

                print("estimates")
                self.v_estimates[:, i] = np.sum(np.multiply(self.reward_z_concat, inner), axis=1)
                print(self.v_estimates)
                input("XXX")

            print("diff")
            self.diff = abs(v_old - self.v_estimates)
            print(np.max(self.diff))
            counter += 1

        plt.plot(self.budget_support, self.v_estimates.transpose())
        plt.show()

        # sys.exit(0)

        return 1, 2


if __name__ == "__main__":

    exp = BudgetValueIterator()
    support, values = exp.run(agent_name='c51')
