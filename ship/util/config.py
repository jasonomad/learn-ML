import os

#################################
#	path and dataset parameter	
#################################

DATA_PATH = 'data'

OUTPUT_DIR = 'output'

TRAIN_FILE = os.path.join(DATA_PATH, 'train.mat')

VAL_FILE = os.path.join(DATA_PATH, 'val.mat')

TEST_FILE = os.path.join(DATA_PATH, 'test.mat')

WEIGHT_DIR = OUTPUT_DIR

#################################
#   hyperparams for learning alg
#################################


BATCH_SIZE = 32

MAX_ITER = 20000

SAVE_ITER = 1000

PRINT_ITER = 100

VALIDATE_ITER = 100

LEARNING_RATE = 0.001

DECAY_STEP = 100

LEARNING_RATE_DECAY = 0.95

STAIRCASE = True

SUMMARY_STEP = 100
#################################
#   
#################################
NUM_CLASS = 1 + 1 # background + class

IMG_INPUT_SIZE = 128

IMG_INPUT_CHANNEL = 1

RANDOM_SEED = 1