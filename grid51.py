#!/usr/bin/env python
from __future__ import print_function

from agents.c51_grid_agent import C51GridAgent, run_all


def test():
    print("XXX")
    grid_agent = C51GridAgent()

    while not grid_agent.is_terminated:
        print("YYY")
        print(grid_agent.run_episode())
        grid_agent.is_terminated = False


if __name__ == "__main__":
    test()
