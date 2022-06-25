# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np
np.set_printoptions(suppress=True)

from game import Game
from funcs import playMatchesBetweenVersions
import loggers as lg
import config


env = Game()

run_version = config.INITIAL_RUN_NUMBER
player1version = -1 # -1이면 사람
player2version = config.INITIAL_MODEL_VERSION
EPISODES = 1
turns_until_tau0 = 0
goes_first = 2  # Bot 먼저

playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, lg.logger_tourney, turns_until_tau0, goes_first)
