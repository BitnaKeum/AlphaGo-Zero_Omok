# -*- coding: utf-8 -*-
# %matplotlib inline

# 게임 규칙을 로드
# 3단계 알고리즘 반복
#    1. Self-play
#    2. 뉴럴 네트워크 재학습
#    3. 뉴럴 네트워크 평가

# run.ipynb의 두번째 셀과 동일
# 실행하면 모든 모델과 메모리 파일은 run 폴더에 저장됨
# 이어서 학습 가능

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload
from game import Game
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches
import config
import loggers as lg
import pickle
# from keras.utils import plot_model



lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

# If loading an existing neural network, copy the config file to root
if config.INITIAL_RUN_NUMBER != None:
    copyfile(config.run_archive_folder  + env.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(4) + '/config.py', './config.py')

######## LOAD MEMORIES IF NECESSARY ########
if config.INITIAL_MEMORY_VERSION == None:
    memory = Memory(config.MEMORY_SIZE)
else:
    print('LOADING MEMORY VERSION ' + str(config.INITIAL_MEMORY_VERSION) + '...')
    memory = pickle.load( open(config.run_archive_folder + env.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(4) + "/memory/memory" + str(config.INITIAL_MEMORY_VERSION).zfill(4) + ".p",   "rb" ) )

######## LOAD MODEL IF NECESSARY ########
# create an untrained neural network objects from the config file
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) +  env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

# If loading an existing neural netwrok, set the weights from that model
if config.INITIAL_MODEL_VERSION != None:
    best_player_version = config.INITIAL_MODEL_VERSION
    print('LOADING MODEL VERSION ' + str(config.INITIAL_MODEL_VERSION) + '...')
    m_tmp = best_NN.read(env.name, config.INITIAL_RUN_NUMBER, best_player_version)
    current_NN.model.set_weights(m_tmp.get_weights())
    best_NN.model.set_weights(m_tmp.get_weights())
# otherwise just ensure the weights on the two players are the same
else:
    best_player_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())

#copy the config file to the run folder
copyfile('./config.py', config.run_folder + 'config.py')
# plot_model(current_NN.model, to_file=config.run_folder + 'models/model.png', show_shapes = True)

print('\n')



######## CREATE THE PLAYERS ########

# best_player: 최고 성능의 신경망을 포함하며 Self-play 메모리를 생성하는 데 사용됨
# current_player: Self-play 메모리에 대한 신경망을 다시 학습시킨 다음 best_player와 경쟁
best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
# user_player = User('player1', env.state_size, env.action_size)

iteration = 0
while True:

    iteration += 1
    reload(lg)
    reload(config)
    print('ITERATION NUMBER ' + str(iteration))

    lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
    print('BEST PLAYER VERSION ' + str(best_player_version))

    ######## Self-play ########
    print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, lg.logger_main, turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=memory)
    print('\n')

    memory.clear_stmemory()

    if len(memory.ltmemory) >= config.MEMORY_SIZE:

        ######## Neural Network re-training ########
        print('NN re-training...')
        current_player.replay(memory.ltmemory)
        print('')

        if iteration % 2 == 0:
            pickle.dump(memory, open(config.run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb"))

        lg.logger_memory.info('====================')
        lg.logger_memory.info('NEW MEMORIES')
        lg.logger_memory.info('====================')

        memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))
        for s in memory_samp:
            current_value, current_probs, _ = current_player.get_preds(s['state'])
            best_value, best_probs, _ = best_player.get_preds(s['state'])

            lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
            lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
            lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
            lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']]  )
            lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  current_probs])
            lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  best_probs])
            lg.logger_memory.info('ID: %s', s['state'].id)
            lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))

            s['state'].render(lg.logger_memory)

        ######## current_player와 best_player 경쟁 ########
        print('TOURNAMENT...')
        scores, _, points, sp_scores = playMatches(best_player, current_player, config.EVAL_EPISODES, lg.logger_tourney, turns_until_tau0=0, memory=None)
        print('\nSCORES')
        print(scores)
        print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
        print(sp_scores)
        print('\n\n')

        if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            best_player_version = best_player_version + 1
            best_NN.model.set_weights(current_NN.model.get_weights())
            best_NN.write(env.name, best_player_version)    # 모델 저장

    else:
        print('MEMORY SIZE: ' + str(len(memory.ltmemory)))
