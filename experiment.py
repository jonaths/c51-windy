
#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from agents.c51_grid_agent import C51GridAgent
from agents.safe_grid_agent import SafeGridAgent
from agents.risky_grid_agent import RiskyGridAgent


num_actions = 4
num_states = 64
max_episodes = 10000
reps = 2

reward_results = np.zeros((reps, max_episodes))
steps_results = np.zeros((reps, max_episodes))
end_state_results = np.zeros((reps, max_episodes))
q_table = np.zeros((reps, num_states, num_actions))
policy = np.zeros((reps, num_states))

for rep in range(reps):

    print("Rep: " + str(rep) + "=========================================================")
    agent = C51GridAgent()

    final_state_buffer, reward_buffer, steps_buffer = [], [], []
    GAME = 0
    t = 0
    r_t = 0

    while GAME < max_episodes:

        print("Game: " + str(GAME) + "---------------------------------------------------")

        result = agent.run_episode(4)

        # if GAME > max_games * 0.8:
        #     agent.setEpsilon(0.0)

        reward_results[rep][GAME] = result['sum_reward']
        steps_results[rep][GAME] = len(result['step_seq'])
        end_state_results[rep][GAME] = result['step_seq'][-1]

        t = 0
        r_t = 0
        GAME += 1

    np.save('statistics/reward.npy', np.array(reward_results))
    np.save('statistics/step.npy', np.array(steps_results))
    np.save('statistics/end_state.npy', np.array(end_state_results))

    states = range(0, num_states)

print("END")

