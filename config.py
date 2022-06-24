#### SELF PLAY
EPISODES = 25
MCTS_SIMS = 50
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
	{'filters': 75, 'kernel_size': (4,4)},
	{'filters': 75, 'kernel_size': (4,4)},
	{'filters': 75, 'kernel_size': (4,4)},
	{'filters': 75, 'kernel_size': (4,4)},
	{'filters': 75, 'kernel_size': (4,4)},
	{'filters': 75, 'kernel_size': (4,4)}
]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.1

# 불러올 모델/메모리 체크포인트 번호
INITIAL_RUN_NUMBER = None
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION =  None

# models 폴더와 memory 폴더를 저장할 상위 폴더
run_folder = './run/'
run_archive_folder = './run_archive/'