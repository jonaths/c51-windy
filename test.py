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

while True:
    env.render()
    action_idx = input()
    print(env.step(action_idx))

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