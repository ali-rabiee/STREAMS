########## Test ############
modelPath = "./models/STREAMS_bs64_ss4_rb30000_gamma0.99_decaylf60000_lr0.001.pt"
MODE = "STREAM" # can be "STREAM" or "Manual" - in Manual mode we do not use STREAM to adjust trajectories


########### TRAIN ##########

MODEL_NAME = 'STREAMS_2'

# Declare number of total episodes
num_episodes = 100000      #[25000,100000,500000] 
logfilename = 'learnDQN.log'

# Hyperparameters, search for different combinations
BATCH_SIZE = 64 
GAMMA = 0.99

EPS_START = 0.99 # Max of exploration rate
EPS_END = 0.2   # min of exploration rate
EPS_DECAY_LAST_FRAME = 60000 # last episode to have decay in exploration rate

TARGET_UPDATE = 1000
LEARNING_RATE = 1e-3
REPLAY_BUFFER_SIZE = 30000
Stack_Size = 4