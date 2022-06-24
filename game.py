import numpy as np
from random import sample


class Game:

	def __init__(self):		
		self.currentPlayer = 1
		self.gameState = GameState(np.array([0] * 7 * 7, dtype=np.int), 1) # 7x7
		self.actionSpace = np.array([0] * 7 * 7, dtype=np.int) # 7x7
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.grid_shape = (7, 7)
		self.input_shape = (2, 7, 7)
		self.name = 'omok'
		self.state_size = len(self.gameState.binary)
		self.action_size = len(self.actionSpace)

	def reset(self):
		self.gameState = GameState(np.array([0]*7*7, dtype=np.int), 1)	# 7x7
		self.currentPlayer = 1
		return self.gameState

	def step(self, action):
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		self.currentPlayer = -self.currentPlayer
		info = None
		return ((next_state, value, done, info))

	def identities(self, state, actionValues):
		identities = [(state, actionValues)]

		currentBoard = state.board
		currentAV = actionValues

		currentBoard = np.array([
			currentBoard[6], currentBoard[5], currentBoard[4], currentBoard[3], currentBoard[2], currentBoard[1], currentBoard[0],
			currentBoard[13], currentBoard[12], currentBoard[11], currentBoard[10], currentBoard[9], currentBoard[8], currentBoard[7],
			currentBoard[20], currentBoard[19], currentBoard[18], currentBoard[17], currentBoard[16], currentBoard[15], currentBoard[14],
			currentBoard[27], currentBoard[26], currentBoard[25], currentBoard[24], currentBoard[23], currentBoard[22], currentBoard[21],
			currentBoard[34], currentBoard[33], currentBoard[32], currentBoard[31], currentBoard[30], currentBoard[29], currentBoard[28],
			currentBoard[41], currentBoard[40], currentBoard[39], currentBoard[38], currentBoard[37], currentBoard[36], currentBoard[35],
			currentBoard[48], currentBoard[47], currentBoard[46], currentBoard[45], currentBoard[44], currentBoard[43], currentBoard[42]
		])

		currentAV = np.array([
			currentAV[6], currentAV[5], currentAV[4], currentAV[3], currentAV[2], currentAV[1], currentAV[0],
			currentAV[13], currentAV[12], currentAV[11], currentAV[10], currentAV[9], currentAV[8], currentAV[7],
			currentAV[20], currentAV[19], currentAV[18], currentAV[17], currentAV[16], currentAV[15], currentAV[14],
			currentAV[27], currentAV[26], currentAV[25], currentAV[24], currentAV[23], currentAV[22], currentAV[21],
			currentAV[34], currentAV[33], currentAV[32], currentAV[31], currentAV[30], currentAV[29], currentAV[28],
			currentAV[41], currentAV[40], currentAV[39], currentAV[38], currentAV[37], currentAV[36], currentAV[35],
			currentAV[48], currentAV[47], currentAV[46], currentAV[45], currentAV[44], currentAV[43], currentAV[42]
		])

		identities.append((GameState(currentBoard, state.playerTurn), currentAV))

		return identities


class GameState():
	def __init__(self, board, playerTurn):
		self.board = board
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.winners = [	# 연속된 5개의 돌이 만들어지는 모든 케이스의 index
			[0,1,2,3,4],
			[1,2,3,4,5],
			[2,3,4,5,6],
			[7,8,9,10,11],
			[8,9,10,11,12],
			[9,10,11,12,13],
			[14,15,16,17,18],
			[15,16,17,18,19],
			[16,17,18,19,20],
			[21,22,23,24,25],
			[22,23,24,25,26],
			[23,24,25,26,27],
			[28,29,30,31,32],
			[29,30,31,32,33],
			[30,31,32,33,34],
			[35,36,37,38,39],
			[36,37,38,39,40],
			[37,38,39,40,41],
			[42,43,44,45,46],
			[43,44,45,46,47],
			[44,45,46,47,48],

			[0,7,14,21,28],
			[7,14,21,28,35],
			[14,21,28,35,42],
			[1,8,15,22,29],
			[8,15,22,29,36],
			[15,22,29,36,43],
			[2,9,16,23,30],
			[9,16,23,30,37],
			[16,23,30,37,44],
			[3,10,17,24,31],
			[10,17,24,31,38],
			[17,24,31,38,45],
			[4,11,18,25,32],
			[11,18,25,32,39],
			[18,25,32,39,46],
			[5,12,19,26,33],
			[12,19,26,33,40],
			[19,26,33,40,47],
			[6,13,20,27,34],
			[13,20,27,34,41],
			[20,27,34,41,48],

			[2,10,18,26,34],
			[1,9,17,25,33],
			[0,8,16,24,32],
			[7,15,23,31,39],
			[8,16,24,32,40],
			[9,17,25,33,41],
			[14,22,30,38,46],
			[15,23,31,39,47],
			[16,24,32,40,48],

			[4,10,16,22,28],
			[5,11,17,23,29],
			[6,12,18,24,30],
			[11,17,23,29,35],
			[12,18,24,30,36],
			[13,19,25,31,37],
			[18,24,30,36,42],
			[19,25,31,37,43],
			[20,26,32,38,44]
		]
		self.playerTurn = playerTurn
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = self._allowedActions()
		self.isEndGame = self._checkForEndGame()
		self.value = self._getValue()
		self.score = self._getScore()

	# 다음에 취할 수 있는 action들을 반환
	def _allowedActions(self):
		allowed = []

		# 상대가 다음 턴에 4개 연속 돌을 만들 수 있는지를 체크해서 막음
		# self.winners에서 10개를 랜덤으로 뽑아서 확인 (모든 케이스를 다 체크하면 너무 오래 걸림)
		rand_idxes = sample(range(0, len(self.winners)), 10)	
		for idx in rand_idxes:
			a, b, c, d, e = self.winners[idx]
			a_val, b_val, c_val, d_val, e_val = self.board[a], self.board[b], self.board[c], self.board[d], self.board[e]

			if a_val == 0 and b_val == -self.playerTurn and c_val == -self.playerTurn and d_val == -self.playerTurn and e_val == 0:
				allowed.append(a)
				allowed.append(e)
			elif a_val == 0 and b_val == -self.playerTurn and c_val == 0 and d_val == -self.playerTurn and e_val == -self.playerTurn:
				allowed.append(c)
			elif a_val == -self.playerTurn and b_val == -self.playerTurn and c_val == 0 and d_val == -self.playerTurn and e_val == 0:
				allowed.append(c)

		# 위 케이스를 만족안하면 돌이 놓여있지 않은 모든 action을 allow
		if len(allowed) == 0:
			for i in range(len(self.board)): # i: 0 ~ 48
				if self.board[i] == 0:	# 놓인 돌이 없으면 allowed
					allowed.append(i)

		return allowed

	def _binary(self):

		currentplayer_position = np.zeros(len(self.board), dtype=np.int)
		currentplayer_position[self.board==self.playerTurn] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-self.playerTurn] = 1

		position = np.append(currentplayer_position,other_position)

		return (position)

	def _convertStateToId(self):
		player1_position = np.zeros(len(self.board), dtype=np.int)
		player1_position[self.board==1] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-1] = 1

		position = np.append(player1_position,other_position)

		id = ''.join(map(str,position))

		return id

	def _checkForEndGame(self):
		if np.count_nonzero(self.board) == 7*7:	# 모든 칸에 돌이 있으면 게임 종료
			return 1

		# 현재 턴인 사람이 5개의 연속된 돌을 만든 경우 게임 종료
		for a,b,c,d,e in self.winners:
			if (self.board[a] + self.board[b] + self.board[c] + self.board[d] + self.board[e] == 5 * -self.playerTurn):
				return 1
		return 0

	def _getValue(self):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose
		for a,b,c,d,e in self.winners:
			if (self.board[a] + self.board[b] + self.board[c] + self.board[d] + self.board[e] == 5 * -self.playerTurn):
				return (-1, -1, 1)
		return (0, 0, 0)


	def _getScore(self):
		tmp = self.value
		return (tmp[1], tmp[2])


	def takeAction(self, action):
		# action: 0 ~ 48
		# playerTurn: -1 or 1
		newBoard = np.array(self.board)
		newBoard[action] = self.playerTurn
		
		newState = GameState(newBoard, -self.playerTurn)

		value = 0
		done = 0

		if newState.isEndGame:	# 모든 칸에 돌이 있거나, 현재 턴인 사람이 이긴 경우
			value = newState.value[0]
			done = 1

		return (newState, value, done) 


	def render(self, logger):
		for r in range(7):
			logger.info([self.pieces[str(x)] for x in self.board[7*r : (7*r + 7)]])
		logger.info('--------------')