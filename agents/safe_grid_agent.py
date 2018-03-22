#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import skimage as skimage
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
from time import sleep
import sys

from lib.c51 import C51Agent

from lib.networks import Networks

import gym
import gym_windy


class SafeGridAgent:
    """
    Un agente que usa c51 para resolver un windy world de 3 x 3
    """

    def __init__(self):

        self.env = gym.make("windy-v0")
        self.env.reset()
        self.is_terminated = False


    def run_episode(self):
        """
        Corre un episodio hasta que ai-gym regresa una bandera done=True.
        En ese momento se setea is_finished y se guardan estadisticas.
        :return:
        """
        self.actions = [2, 2, 1, 1, 0, 0]

        while not self.is_terminated:

            # sleep(0.1)
            # self.env.render()
            self.action_idx = self.actions.pop(0)
            _, _, self.done, self.misc = self.env.step(self.action_idx)
            self.is_terminated = self.done

        self.env.reset()
        self.is_terminated = False

        return self.misc


if __name__ == "__main__":
    pass
