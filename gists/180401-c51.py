#!/usr/bin/env python
from __future__ import print_function

import math
import random
from collections import deque

import sys
from plot_histogram import PlotHistogramRT

import numpy as np


def reduce_noise(bins, reference=0.75):
    """
    Substracts from each bin in c51 a value that corresponds to the value where
    an histogram contains 73% of the largest values.
    :param reference:
    :param bins:
    :return:
    """
    final_bins = []

    for local_bin in bins:
        # creates an histogram with 20 bins
        counts, bin_edges = np.histogram(local_bin, bins=20)
        # calculates the threshold
        threshold = reference * np.sum(counts)
        # finds the upper bin value
        val = bin_edges[np.argmax(np.cumsum(counts) > threshold)]
        # subtract the val  (noise) to all c51 bins ensuring that is no less
        # than 0
        local_bin = np.asarray(local_bin - val).clip(min=0)
        # distributes the remaining bins proportionally to their size
        local_sum = np.sum(local_bin)
        local_bin = local_bin / local_sum
        final_bins.append(local_bin)
    return final_bins


class C51Agent:
    def __init__(self, state_size, action_size, num_atoms):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the DQN
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1
        self.initial_epsilon = 1.0
        # self.initial_epsilon = 0.0001
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 1000
        self.explore = 200000
        # self.observe = 1
        # self.explore = 2
        self.frame_per_action = 4
        self.update_target_freq = 2000
        self.timestep_per_train = 500  # Number of timesteps between training interval

        # Initialize Atoms
        self.num_atoms = num_atoms  # 51 for C51
        self.v_max = +15  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = -15  # -0.1*26 - 1 = -3.6
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Create replay memory using deque
        self.memory = deque()
        self.max_memory = 1000  # number of previous transitions to remember

        # Models for value distribution
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size = 100  # window size for computing rolling statistics
        self.mavg_reward = []  # Moving average of real reward
        self.var_reward = []
        self.mavg_steps = []  # Moving Average of Survival Time
        self.var_steps = []  # Variance of Survival Time
        self.end_count = []
        self.total_reward = 0
        self.real_time_plotter = PlotHistogramRT(action_size, num_atoms, ['<-', '->'])

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        # if np.random.rand() <= self.epsilon:
        if np.random.rand() <= 0:
            print("----------Random Action----------")
            action_idx = random.randrange(self.action_size)
        else:
            action_idx = self.get_optimal_action(state)

        return action_idx

    def get_optimal_action(self, state, b=4.0, rp=0.1):
        """Get optimal action for a state
        """
        z_concat, q = self.predict(state)

        # Pick action with the biggest Q value
        action_idx = np.argmax(q)

        print(z_concat)
        print(self.z)
        print(len(self.z))
        print(q)

        index = np.searchsorted(self.z, -b) - 1
        print("aa")
        print(index)
        prob_dead_given_b = np.cumsum(z_concat, 1)[:, index]
        print(prob_dead_given_b)
        prob_dead_given_b = prob_dead_given_b - rp
        print(prob_dead_given_b)
        prob_dead_given_b = prob_dead_given_b / np.sum(prob_dead_given_b)
        print(prob_dead_given_b)

        prob_alive_given_b = 1 - prob_dead_given_b
        weighted_q = prob_alive_given_b * q
        print(weighted_q)

        sys.exit(0)

        return action_idx

    # def get_optimal_action(self, state):
    #     """Get optimal action for a state
    #     """
    #     z_concat, q = self.predict(state)
    #
    #     # Pick action with the biggest Q value
    #     action_idx = np.argmax(q)
    #
    #     return action_idx

    def predict(self, state):
        z = self.model.predict(state)  # Return a list [1x51, 1x51, 1x51]
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return reduce_noise(z_concat), q
        # return z_concat, q

    def plot_histogram(self, state, verbose=True):
        z_concat, q = self.predict(state)
        if verbose:
            print("=q")
            print(q)
            action_idx = np.argmax(q)
            print("=q_arg_max")
            print(action_idx)
        self.real_time_plotter.update(np.array(self.z), z_concat)



    def shape_reward(self, r_t, misc, prev_misc, t, is_terminated):

        # # Check any kill count
        # if (misc[0] > prev_misc[0]):
        #     r_t = r_t + 1
        #
        # if (misc[1] < prev_misc[1]):  # Use ammo
        #     r_t = r_t - 0.1
        #
        # if (misc[2] < prev_misc[2]):  # Loss HEALTH
        #     r_t = r_t - 0.1
        #
        # if(is_terminated):
        #     r_t = r_t - 5

        self.total_reward = self.total_reward + r_t
        return r_t

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        state_inputs = np.zeros(((num_samples,) + self.state_size))
        next_states = np.zeros(((num_samples,) + self.state_size))
        m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(self.action_size)]
        action, reward, done = [], [], []

        for i in range(num_samples):
            state_inputs[i, :, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            next_states[i, :, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        z = self.model.predict(next_states)  # Return a list [32x51, 32x51, 32x51]
        z_ = self.target_model.predict(next_states)  # Return a list [32x51, 32x51, 32x51]

        # Get Optimal Actions for the next states (from distribution z)
        optimal_action_idxs = []
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)  # length (num_atoms x num_actions)
        q = q.reshape((num_samples, self.action_size), order='F')
        optimal_action_idxs = np.argmax(q, axis=1)

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

        loss = self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, nb_epoch=1, verbose=0)

        return loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)