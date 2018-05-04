#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

# from agents.c51_grid_agent import C51GridAgent
# from agents.safe_grid_agent import SafeGridAgent
# from agents.risky_grid_agent import RiskyGridAgent


# pseudocodigo
# definir b[]
# definir thres
# definir maximo numero de episodios
# definir v inicial
# para cada presupuesto
# from agents.c51_grid_agent import C51GridAgent


class BudgetValueIterator:
    """
    Runs an experiment until Done is set to true by ai gym using an agent
    """

    def __init__(self, reward_z_concat, reward_support):
        self.v_estimates = None
        self.budget_support = None
        self.reward_support = None
        self.reward_z_concat = None
        self.diff = None
        self.gamma = 0.95
        self.reward_z_concat = reward_z_concat
        self.reward_support = reward_support

    def save_plot(self, counter, show=False):
        plt.plot(self.budget_support[1:], self.v_estimates[:,1:].transpose())
        plt.ylabel('Expected Value')
        plt.xlabel('Budget')
        plt.savefig('value_plots/' + str(counter) + '_value.png')
        if show:
            plt.show()
        plt.clf()

    def run(self):

        # en -1 acumula lo que no esta dentro del soporte
        # por eso el for empieza desde el segundo indice
        self.budget_support = np.arange(-15, 16, 1)
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

        # input("Continue... ")

        counter = 0
        while np.max(self.diff) > 0.05:
            print("================================================")
            print(counter)
            v_old = np.copy(self.v_estimates)
            # empieza desde el segundo indice para acumular lo que no esta dentro del
            # soporte
            for i in range(0, len(self.budget_support)):
                print("--------------------------------------------")
                print(i, self.budget_support[i], self.reward_support)

                # lo asigna al entero inmediato inferior
                print("b_next")
                b_next = np.floor(self.reward_support + self.budget_support[i])
                print(b_next)

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

                # print("v_next")
                # v_next = self.v_estimates[:, next_indices]
                # print(v_next)

                print("v_next_max")
                v_next = np.max(self.v_estimates[:, next_indices], axis=0)
                v_next[b_next < 0] = 0
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
                # input("XXX")

            print("diff")
            self.diff = abs(v_old - self.v_estimates)
            print(np.max(self.diff))

            # if counter in [0, 1, 2, 3, 4]:
            #     self.save_plot(counter)

            counter += 1

        self.save_plot(counter, show=True)

        # sys.exit(0)

        return self.budget_support, self.v_estimates


if __name__ == "__main__":
    pass
    #
    # retrieve learned model predictions
    # agent = C51GridAgent()
    # agent.reset()
    # reward_z_concat, q, _, _ = agent.agent.predict(agent.s_t)
    # reward_support = np.array(agent.agent.z)

    # para probar
    # exp.reward_z_concat = np.array([
    #     [0.1, 0.1, 0.1, 0.20, 0.5],
    #     [0.1, 0.2, 0.2, 0.5, 0.0]
    # ])
    # exp.reward_support = np.array(
    #     [-2, -1, 0, 1, 2]
    # )

    exp = BudgetValueIterator(reward_z_concat, reward_support)

    support, values = exp.run()
