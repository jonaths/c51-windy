#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import skimage as skimage
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
import sys
from time import sleep

from helpers.c51 import C51Agent
from plotters import policy_plotter

from helpers.networks import Networks

import gym
import gym_windy
from rms.rms import RmsAlg

from val_tester_new import BudgetValueIterator


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

        self.env = gym.make("beach-v0")
        self.env.reset()

        # misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        self.misc = {'final_state': None}
        self.prev_misc = self.misc

        self.action_size = 4
        self.img_rows, self.img_cols = 8, 8
        self.img_channels = 2

        # C51
        self.num_atoms = 51

        self.state_size = (self.img_rows, self.img_cols, self.img_channels)
        self.agent = C51Agent(self.state_size, self.action_size, self.num_atoms)

        self.agent.model = Networks.value_distribution_network(
            self.state_size, self.num_atoms,
            self.action_size,
            self.agent.learning_rate)
        # self.agent.load_model("models/c51_ddqn.h5")
        self.agent.target_model = Networks.value_distribution_network(
            self.state_size, self.num_atoms,
            self.action_size,
            self.agent.learning_rate)

        self.data = range(self.img_rows * self.img_cols)

        self.init_obs = 41
        self.x_t = np.reshape(to_categorical(self.data)[self.init_obs],
                              (self.img_rows, self.img_cols))
        self.s_t = np.stack(([self.x_t] * self.img_channels), axis=0)
        self.s_t = np.rollaxis(self.s_t, 0, 3)
        self.s_t = np.expand_dims(self.s_t, axis=0)
        self.s_t1 = self.s_t

        self.is_terminated = False

        # Start training
        self.epsilon = self.agent.initial_epsilon
        self.GAME = 0
        self.t = 0
        # Maximum episode life (Proxy for agent performance)
        self.max_life = 0
        self.life = 0
        self.state = ""

        # Buffer to compute rolling statistics
        self.final_state_buffer, self.reward_buffer, self.steps_buffer = [], [], []

        # RmsAlg(rthres, influence, risk_default)
        self.rms = RmsAlg(rthres=-1, influence=2, risk_default=0)
        self.rms.add_to_v(self.init_obs, self.env.ind2coord(self.init_obs))

    def process_state(self, obs):
        """
        Converts an integer state as provided by ai-gym into a numpy matrix state
        required by the learning algorithm.
        :param obs:
        :return:
        """
        x_t1 = np.reshape(to_categorical(self.data)[obs],
                          (self.img_rows, self.img_cols))
        x_t1 = np.reshape(x_t1, (1, self.img_rows, self.img_cols, 1))

        # considera historial en un canal
        # self.s_t1 = np.append(self.x_t1, self.s_t[:, :, :, :1], axis=3)

        # no considera historial en un canal
        s_t1 = np.append(x_t1, x_t1, axis=3)

        return s_t1

    def plot_policy(self, int_states=[2]):
        """
        Makes plot policies for all actions and states  int_states
        :return:
        """
        predictions = []
        # qs = []
        # budget_qs = []
        # probs_of_alive = []
        vi = []
        for s in range(len(int_states)):
            # creates a prediction to plot policies
            prediction = self.agent.predict(self.process_state(int_states[s]))
            predictions.append(prediction[0])
            # qs.append(prediction[1])
            # budget_qs.append(prediction[2])
            # probs_of_alive.append(prediction[3])
            exp = BudgetValueIterator(np.array(prediction[0]), np.array(self.agent.z))
            budget_support, values = exp.run()
            vi.append(values)
            # fills arrays
            # predictions = np.array(predictions)
            # qs = np.array(qs)
            # probs_of_alive = np.array(probs_of_alive)
            # budget_qs = np.array(budget_qs)
            # policy_plotter.plot(int_states, np.array(budget_support), np.array(vi))
            # policy_plotter.plot(int_states, self.agent.z, budget_qs)
            # self.agent.plot_policy(int_states, budget_qs)

    def reset(self):
        self.data = range(self.img_rows * self.img_cols)
        self.x_t = np.reshape(to_categorical(self.data)[self.init_obs], (self.img_rows, self.img_cols))
        self.x_t = np.reshape(to_categorical(self.data)[2], (self.img_rows, self.img_cols))
        self.s_t = np.stack(([self.x_t] * self.img_channels), axis=0)
        self.s_t = np.rollaxis(self.s_t, 0, 3)
        self.s_t = np.expand_dims(self.s_t, axis=0)

        self.is_terminated = False

    def run_episode(self, b=0):
        """
        Corre un episodio hasta que ai-gym regresa una bandera done=True.
        En ese momento se setea is_finished y se guardan estadisticas.
        :return:
        """

        # inicializacion de misc

        self.misc = {'sum_reward': 0}

        # self.plot_policy()

        while not self.is_terminated:

            current_budget = b + self.misc['sum_reward']
            # current_budget = b + 0

            # print("current budget")
            # print(current_budget)

            self.loss = 0
            self.r_t = 0
            self.a_t = np.zeros([self.action_size])

            # sleep(0.1)
            # print("st")
            # print(self.s_t)
            # self.env.render()
            # self.agent.plot_histogram(self.s_t1)
            # input("XXX")

            # self.action_idx = input("action")
            self.action_idx, add_info = self.agent.get_action(self.s_t, current_budget)

            self.a_t[self.action_idx] = 1
            self.obs, self.r_t, self.done, self.misc = self.env.step(self.action_idx)
            self.is_terminated = self.done

            self.rms.update(
                s=self.misc['step_seq'][-2],
                r=self.r_t,
                sprime=self.obs,
                sprime_features=self.env.ind2coord(self.obs))

            risk_penalty = self.rms.get_risk(self.obs)
            self.r_t = self.r_t + risk_penalty

            if self.is_terminated:

                if self.life > self.max_life:
                    self.max_life = self.life

                self.GAME += 1
                self.final_state_buffer.append(
                    1 if self.misc['step_seq'][-1] in self.env.hole_state else 0
                )
                self.steps_buffer.append(len(self.misc['step_seq']))
                self.reward_buffer.append(self.misc['sum_reward'])
                self.env.reset()

            self.s_t1 = self.process_state(self.obs)

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



if __name__ == "__main__":
    pass
