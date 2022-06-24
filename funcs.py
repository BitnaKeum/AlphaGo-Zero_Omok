import numpy as np
import random

import loggers as lg

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config
import time


def playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0, goes_first = 0):
    # run_version: the run version number where the computer player is located
    # player1version: the version number of the first player (-1 for human)
    # player2version: the version number of the second player (-1 for human)
    # EPISODES: how many games to play
    # logger: where to log the game to
    # turns_until_tau0: ?
    # goes_first: which player to go first (0 for random)

    # player1과 player2 설정

    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_NN.read(env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())   
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
        
        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first, mode='test')

    return (scores, memory, points, sp_scores)


def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory=None, goes_first=0, mode='train'):
    # player1: first player agent
    # player2: second player agent
    # logger: where to log the game to
    # turns_until_tau0: ?
    # goes_first: which player to go first (0 for random)

    env = Game()
    scores = {player1.name:0, "drawn": 0, player2.name:0}
    sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
    points = {player1.name: [], player2.name: []}

    for e in range(EPISODES):
        if mode == 'train':
            logger.info('====================')
            logger.info('EPISODE %d OF %d', e+1, EPISODES)
            logger.info('====================')

        state = env.reset()

        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        # 선공 선택
        if goes_first == 0:
            player1Starts = random.randint(0, 1) * 2 - 1
        else:
            player1Starts = goes_first

        if player1Starts == 1:
            players = {
                1: {"agent": player1, "name": player1.name},
                -1: {"agent": player2, "name": player2.name}
            }
            logger.info(player1.name + ' plays as X')
            logger.info('--------------')
        else:
            players = {
                1: {"agent": player2, "name": player2.name},
                -1: {"agent": player1, "name": player1.name}
            }
            logger.info(player2.name + ' plays as X')
            logger.info('--------------')

        env.gameState.render(logger)

        # 초기 board 출력
        if mode == 'test':  # test
            print("\nInitial Board")
            for r in range(7):
                print([env.gameState.pieces[str(x)] for x in env.gameState.board[7 * r: (7 * r + 7)]])
            print("\nBot: 'X',  You: 'O'")

        while done == 0:    # 승부날 때까지 반복
            if mode == 'test':  # test
                # test 시에는 player1(사람)의 state.playerTurn == -1

                if state.playerTurn == -1:
                    print("\n\n\n\n\n\n=========== Your Turn ===========\n")
                else:
                    print("\n\n\n\n\n\n=========== Bot's Turn ===========\n")

            turn = turn + 1

            # select action
            if mode == 'test' and state.playerTurn == -1: # test + 사람 Turn
                print("Available actions")
                for r in range(7):  # 입력 가능한 action 번호
                    print("[ ", end="")
                    for idx, x in enumerate(env.gameState.board[7 * r: (7 * r + 7)]):
                        if env.gameState.pieces[str(x)] == '-': # 가능
                            action_num = 7 * r + idx
                            if action_num < 10: # 한자리 띄어쓰기
                                if idx == 6:
                                    print(" " + str(action_num) + "]")
                                else:
                                    print(" " + str(action_num) + ",  ", end="")
                            else:
                                if idx == 6:
                                    print(str(action_num) + "]")
                                else:
                                    print(str(action_num) + ",  ", end="")
                        else:   # 불가능
                            if idx == 6:
                                print(" X]")
                            else:
                                print(" X,  ", end="")
                print("\n")
                    # print([env.gameState.pieces[str(x)] for x in env.gameState.board[7 * r: (7 * r + 7)]])
                action = int(input('Select action (0 ~ 48): '))

            else:   # train 또는 test+봇 Turn
                #### Run the MCTS algo and return an action
                if turn < turns_until_tau0:
                    action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1) # User()
                else:
                    action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0) # Agent()

                if memory != None:
                    #### Commit the move to memory
                    memory.commit_stmemory(env.identities, state, pi)

                logger.info('action: %d', action)
                for r in range(env.grid_shape[0]):
                    logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x,2)) for x in pi[env.grid_shape[1]*r : (env.grid_shape[1]*r + env.grid_shape[1])]])
                logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(MCTS_value,2))
                logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(NN_value,2))
                logger.info('====================')

            ### Do the action
            state, value, done, _ = env.step(action) #the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move
            # player 1이 action을 취해서 이기면 value는 -1 (이게 player -1의 value인건가?)

            # action 한번 수행할 때마다 board 출력
            env.gameState.render(logger)
            if mode == 'test':  # test
                for r in range(7):
                    print([env.gameState.pieces[str(x)] for x in env.gameState.board[7 * r: (7 * r + 7)]])
                time.sleep(2)

            if done == 1:
                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value

                    memory.commit_ltmemory()

                if value == 1:
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    if mode == 'test':  # test
                        if state.playerTurn == -1:
                            print("\nYou WIN !! :)\n")
                        else:
                            print("\nYou LOSE !! :(\n")
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    logger.info('%s WINS!', players[-state.playerTurn]['name'])
                    if mode == 'test':  # test
                        if state.playerTurn == -1:
                            print("\nYou LOSE !! :(\n")
                        else:
                            print("\nYou WIN !! :)\n")
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1

                    if state.playerTurn == 1:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    logger.info('DRAW...')
                    if mode == 'test':  # test
                        print("\nDRAW !!\n")
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])

    return (scores, memory, points, sp_scores)
