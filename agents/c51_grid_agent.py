#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import skimage as skimage
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
import sys
from time import sleep

from lib.c51 import C51Agent

from lib.networks import Networks

import gym
import gym_windy


def preprocessImg(img, size):
    # Cambia el orden. Al final hay 3 porque es RGB
    # It becomes (640, 480, 3)
    img = np.rollaxis(img, 0, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)

    return img


class C51GridAgent:
    """
    Un agente que usa c51 para resolver un windy world de 3 x 3
    """

    def __init__(self):
        # Avoid Tensorflow eats up GPU memory
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)

        self.env = gym.make("windy-v0")
        self.env.reset()

        # misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        self.misc = {'final_state': None}
        self.prev_misc = self.misc

        self.action_size = 2
        self.img_rows, self.img_cols = 1, 5
        self.img_channels = 2

        # C51
        self.num_atoms = 51

        self.state_size = (self.img_rows, self.img_cols, self.img_channels)
        self.agent = C51Agent(self.state_size, self.action_size, self.num_atoms)

        self.agent.model = Networks.value_distribution_network(
            self.state_size, self.num_atoms,
            self.action_size,
            self.agent.learning_rate)
        self.agent.load_model("models/c51_ddqn.h5")
        self.agent.target_model = Networks.value_distribution_network(
            self.state_size, self.num_atoms,
            self.action_size,
            self.agent.learning_rate)

        self.data = range(self.img_rows * self.img_cols)
        self.x_t = np.reshape(to_categorical(self.data)[0], (self.img_rows, self.img_cols))
        self.s_t = np.stack(([self.x_t] * self.img_channels), axis=0)
        self.s_t = np.rollaxis(self.s_t, 0, 3)
        self.s_t = np.expand_dims(self.s_t, axis=0)
        self.s_t1 = self.s_t
        # print(s_t)

        self.is_terminated = False

        # Start training
        self.epsilon = self.agent.initial_epsilon
        self.GAME = 0
        self.t = 0
        self.max_life = 0  # Maximum episode life (Proxy for agent performance)
        self.life = 0
        self.state = ""

        # Buffer to compute rolling statistics
        self.final_state_buffer, self.reward_buffer, self.steps_buffer = [], [], []

    def run_episode(self):
        """
        Corre un episodio hasta que ai-gym regresa una bandera done=True.
        En ese momento se setea is_finished y se guardan estadisticas.
        :return:
        """

        while not self.is_terminated:

            self.loss = 0
            self.r_t = 0
            self.a_t = np.zeros([self.action_size])

            # sleep(0.1)
            print("st:", self.s_t1)
            self.env.render()
            self.agent.plot_histogram(self.s_t1)

            # Epsilon Greedy
            self.action_idx = input("action")
            # self.action_idx = self.agent.get_action(self.s_t)

            self.a_t[self.action_idx] = 1
            self.obs, self.r_t, self.done, self.misc = self.env.step(self.action_idx)
            self.is_terminated = self.done

            if self.is_terminated:

                if self.life > self.max_life:
                    self.max_life = self.life

                self.GAME += 1
                self.final_state_buffer.append(0 if self.misc['step_seq'][-1] == 8 else 1)
                self.steps_buffer.append(len(self.misc['step_seq']))
                self.reward_buffer.append(self.misc['sum_reward'])
                self.env.reset()

            self.x_t1 = np.reshape(to_categorical(self.data)[self.obs], (self.img_rows, self.img_cols))
            self.x_t1 = np.reshape(self.x_t1, (1, self.img_rows, self.img_cols, 1))

            # considera historial en un canal
            # self.s_t1 = np.append(self.x_t1, self.s_t[:, :, :, :1], axis=3)

            # no considera historial en un canal
            self.s_t1 = np.append(self.x_t1, self.x_t1, axis=3)

            self.r_t = self.agent.shape_reward(self.r_t,
                                               self.misc,
                                               self.prev_misc,
                                               self.t,
                                               self.is_terminated)

            if self.is_terminated:
                self.life = 0
            else:
                self.life += 1

            # update the cache
            self.prev_misc = self.misc

            # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
            self.agent.replay_memory(self.s_t,
                                         self.action_idx,
                                         self.r_t, self.s_t1,
                                         self.is_terminated,
                                         self.t)

            # Do the training
            if self.t > self.agent.observe and self.t % self.agent.timestep_per_train == 0:
                self.loss = self.agent.train_replay()

            self.s_t = self.s_t1
            self.t += 1

            # save progress every 10000 iterations
            if self.t % 1000 == 0:
                print("Now we save model")
                self.agent.model.save_weights("models/c51_ddqn.h5", overwrite=True)

            # print info

            if self.t <= self.agent.observe:
                self.state = "observe"
            elif self.agent.observe < self.t <= self.agent.observe + self.agent.explore:
                self.state = "explore"
            else:
                self.state = "train"

            if self.is_terminated:

                print("TIME", self.t, "/ GAME", self.GAME,
                      "/ STATE", self.state, \
                      "/ EPSILON", self.agent.epsilon,
                      "/ ACTION", self.action_idx, "/ REWARD",
                      self.r_t, \
                      "/ LIFE", self.max_life, "/ LOSS",
                      self.loss)

                # Save Agent's Performance Statistics
                if self.GAME % self.agent.stats_window_size == 0 and self.t > self.agent.observe:
                    print("Update Rolling Statistics")
                    self.agent.mavg_reward.append(
                        np.mean(np.array(self.reward_buffer)))
                    self.agent.var_reward.append(
                        np.std(np.array(self.reward_buffer)))
                    self.agent.mavg_steps.append(
                        np.mean(np.array(self.steps_buffer)))
                    self.agent.var_steps.append(
                        np.std(np.array(self.steps_buffer)))
                    self.agent.end_count.append(np.average(
                        np.array(self.final_state_buffer)))

                    # Reset rolling stats buffer
                    self.final_state_buffer, self.reward_buffer, self.steps_buffer = [], [], []

                    # Write Rolling Statistics to file
                    with open("statistics/c51_ddqn_stats.txt",
                              "w") as stats_file:
                        stats_file.write(
                            'Game: ' + str(self.GAME) + '\n')
                        stats_file.write('Max Score= ' + str(
                            self.max_life) + '\n')
                        stats_file.write('mavg_reward= ' + str(
                            self.agent.mavg_reward) + '\n')
                        stats_file.write('var_reward= ' + str(
                            self.agent.var_reward) + '\n')
                        stats_file.write('mavg_steps= ' + str(
                            self.agent.mavg_steps) + '\n')
                        stats_file.write('var_steps= ' + str(
                            self.agent.var_steps) + '\n')
                        stats_file.write('end_count= ' + str(
                            self.agent.end_count) + '\n')

        self.is_terminated = False

        return self.misc


def run_all():
    """
    El programa original.
    :return:
    """
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    env = gym.make("windy-v0")
    env.reset()

    # misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    misc = {'final_state': None}
    prev_misc = misc

    action_size = 4
    img_rows, img_cols = 3, 3
    img_channels = 2

    # C51
    num_atoms = 51

    state_size = (img_rows, img_cols, img_channels)
    agent = C51Agent(state_size, action_size, num_atoms)

    agent.model = Networks.value_distribution_network(
        state_size, num_atoms, action_size,
        agent.learning_rate)
    agent.load_model("models/c51_ddqn.h5")
    agent.target_model = Networks.value_distribution_network(
        state_size, num_atoms, action_size,
        agent.learning_rate)

    data = range(img_rows * img_cols)
    x_t = np.reshape(to_categorical(data)[0],
                     (img_rows, img_cols))
    s_t = np.stack(([x_t] * img_channels), axis=0)
    s_t = np.rollaxis(s_t, 0, 3)
    s_t = np.expand_dims(s_t, axis=0)
    s_t1 = s_t
    # print(s_t)

    is_terminated = False

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics
    final_state_buffer, reward_buffer, steps_buffer = [], [], []

    while not is_terminated:

        loss = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # sleep(0.1)
        env.render()
        agent.plot_histogram(s_t1)

        # Epsilon Greedy
        action_idx = input("action")
        # action_idx = agent.get_action(s_t)
        a_t[action_idx] = 1
        obs, r_t, done, misc = env.step(action_idx)
        is_terminated = done

        if is_terminated:
            if life > max_life:
                max_life = life
            GAME += 1
            final_state_buffer.append(
                0 if misc['step_seq'][-1] == 8 else 1)
            steps_buffer.append(len(misc['step_seq']))
            reward_buffer.append(misc['sum_reward'])
            env.reset()

        x_t1 = np.reshape(to_categorical(data)[obs],
                          (img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :1], axis=3)

        r_t = agent.shape_reward(r_t, misc, prev_misc, t,
                                 is_terminated)

        if is_terminated:
            life = 0
        else:
            life += 1

        # update the cache
        prev_misc = misc

        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1,
                            is_terminated, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            loss = agent.train_replay()

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            agent.model.save_weights("models/c51_ddqn.h5",
                                     overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif agent.observe < t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if is_terminated:

            is_terminated = False
            print("TIME", t, "/ GAME", GAME, "/ STATE",
                  state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION",
                  action_idx, "/ REWARD", r_t, \
                  "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_reward.append(
                    np.mean(np.array(reward_buffer)))
                agent.var_reward.append(
                    np.std(np.array(reward_buffer)))
                agent.mavg_steps.append(
                    np.mean(np.array(steps_buffer)))
                agent.var_steps.append(
                    np.std(np.array(steps_buffer)))
                agent.end_count.append(np.average(
                    np.array(final_state_buffer)))

                # Reset rolling stats buffer
                final_state_buffer, reward_buffer, steps_buffer = [], [], []

                # Write Rolling Statistics to file
                with open("statistics/c51_ddqn_stats.txt",
                          "w") as stats_file:
                    stats_file.write(
                        'Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score= ' + str(
                        max_life) + '\n')
                    stats_file.write('mavg_reward= ' + str(
                        agent.mavg_reward) + '\n')
                    stats_file.write('var_reward= ' + str(
                        agent.var_reward) + '\n')
                    stats_file.write('mavg_steps= ' + str(
                        agent.mavg_steps) + '\n')
                    stats_file.write('var_steps= ' + str(
                        agent.var_steps) + '\n')
                    stats_file.write('end_count= ' + str(
                        agent.end_count) + '\n')


if __name__ == "__main__":
    run_all()
