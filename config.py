# ==== DATASET PATHS
BOP_PATH = "/PATH/TO/bop_toolkit"
LM_PATH = "/PATH/TO/BOP19/lm"
LMO_PATH = "/PATH/TO/BOP19/lmo"
YCBV_PATH = "/PATH/TO/BOP19/ycbv"
POSECNN_LM_RESULTS_PATH = "/PATH/TO/PoseCNN_LINEMOD_6D_results"
POSECNN_YCBV_RESULTS_PATH = "/PATH/TO/YCB_Video_toolbox"

# normals
USE_NORMALS = True
PRECOMPUTE_NORMALS = False
# symmetry
USE_SYMMETRY = True
SYMMETRY_BEST = True and USE_SYMMETRY
SYMMETRY_AXIS_DELTA = 5
assert 360 % SYMMETRY_AXIS_DELTA == 0
USE_CANONICAL = False or USE_SYMMETRY
# physical plausibility
USE_CONTACT = True
SCENE_CONTACT = 'ref' if USE_CONTACT else ''  # ''.. only plane, 'ref'.. refine others too
SCENE_REFINE = SCENE_CONTACT == 'ref' and USE_CONTACT
TOL_CONTACT = 0.01
UPDATE_SOURCE_DISTANCE = True and USE_CONTACT  # False for LM as it has no other targets (and src vs plane is static)
# rendering-based verification
USE_VERIFICATION = True
# segmentation
USE_SEGMENT = True
C_SEGMENT = 7.0

# iterations and replay buffer
BATCH_SIZE = 32
BATCH_SIZE_BUFFER = 32  # consistently use 32 samples per update
ITER_TRAIN, ITER_EVAL = 10, 10
NUM_TRAJ = 4

# agent parameters
EXPERT_MODE = 'steady'
DISENTANGLED = True
STEPSIZES = [0.0033, 0.01, 0.03, 0.09, 0.27]  # trippling

# model parameters
IN_CHANNELS = 6 if USE_NORMALS else 3
IN_CHANNELS += 1 if USE_CONTACT else 0
FEAT_DIM = 2048
STATE_DIM = FEAT_DIM
HEAD_DIM = 256
ACTION_DIM = 6
NUM_ACTIONS = 6  # 6 actions [+-x, +-y, +-z]
NUM_STEPSIZES = len(STEPSIZES)  # 5 step sizes
NUM_NOPS = 3  # for all actions

# RL parameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
CLIP_VALUE = False
C_VALUE, C_ENTROPY = 0.3, 1e-3
