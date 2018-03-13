import gym
import gym_windy
from keras.utils import to_categorical
# from numpy import array
# from numpy import argmax
import numpy as np


# action_size = 4
# img_rows, img_cols = 4, 3
# img_channels = 2
# num_atoms = 51
#
# state_size = (img_rows, img_cols, img_channels)
#
# data = range(img_rows * img_cols)
# x_t = np.reshape(to_categorical(data)[0], (img_rows, img_cols))
# s_t = np.stack(([x_t] * img_channels), axis=0)
# s_t = np.rollaxis(s_t, 0, 3)
# s_t = np.expand_dims(s_t, axis=0)
#
# print s_t.shape
# print(s_t)


# print(s_t)
# print(s_t.shape)

env = gym.make("windy-v0")
env.reset()

# call this to be able to use render() method

# top
actions = [2, 1, 1, 0]

# middle
# actions = [2, 2, 1, 1, 0, 0]

# bottom
# actions = [2, 2, 2, 1, 1, 0, 0, 0]

# Buffer to compute rolling statistics
final_state_buffer, reward_buffer, steps_buffer = [], [], []

t = 0
i = 0
GAME = 0
mavg_reward, var_reward, mavg_steps, var_steps, end_count = [], [], [], [], []

while GAME < 10000:

    # env.render()
    obs, r_t, done, misc = env.step(actions[i])
    is_terminated = done
    print(t, actions[i], obs)
    i += 1
    t += 1

    if is_terminated:
        final_state_buffer.append(0 if misc['step_seq'][-1] == 8 else 1)
        steps_buffer.append(len(misc['step_seq']))
        reward_buffer.append(misc['sum_reward'])
        env.reset()
        i = 0
        GAME += 1
        print("=== DONE")

        if GAME % 50 == 0:
            print("Update Rolling Statistics")
            mavg_reward.append(
                np.mean(np.array(reward_buffer)))
            var_reward.append(
                np.std(np.array(reward_buffer)))
            mavg_steps.append(
                np.mean(np.array(steps_buffer)))
            var_steps.append(
                np.std(np.array(steps_buffer)))
            end_count.append(
                np.average(np.array(final_state_buffer)))

            # Reset rolling stats buffer
            final_state_buffer, reward_buffer, steps_buffer = [], [], []

            # Write Rolling Statistics to file
            with open("statistics/domain_stats.txt",
                      "w") as stats_file:
                stats_file.write('Game: ' + str(GAME) + '\n')
                stats_file.write('mavg_reward: ' + str(
                    mavg_reward) + '\n')
                stats_file.write('var_reward: ' + str(
                    var_reward) + '\n')
                stats_file.write('mavg_steps: ' + str(
                    mavg_steps) + '\n')
                stats_file.write(
                    'var_steps: ' + str(var_steps) + '\n')
                stats_file.write(
                    'end_count: ' + str(end_count) + '\n')

# action_idx = input()
# print(env.step(action_idx))
#
# action_idx = input()
# print(env.step(action_idx))
#
# action_idx = input()
# print(env.step(action_idx))
#
# action_idx = input()
# print(env.step(action_idx))

# print()
# env.render()
# print(env.step(2))
# env.render()
# #print(env.step(2))
# print(env.step(1))
# env.render()
# print(env.step(1))
# env.render()
# print(env.step(0))
# env.render()
# print(env.step(0))
# env.render()
# print(env.step(0))
# env.render()
# print(env.step(0))
# env.render()
#
# print(env.action_space)
# env.action_space.sample()
#

# define example
# data = range(12)
# print(data)
# # one hot encode
# encoded = to_categorical(data)
# print(encoded)
# # invert encoding
# inverted = argmax(encoded[0])
# print(inverted)