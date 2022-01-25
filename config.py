import torch
from PIL import Image

LOAD_MODEL = False
SAVE_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC1 = "disc1.pth.tar"
CHECKPOINT_DISC2 = "disc2.pth.tar"
CHECKPOINT_JT = "jt.pth.tar"
MODEL_DISC2 = "D:\\IIT BHU\\Research\\CWNU\\Joint-SRVDNet\\model_saves\\Discriminator_2"
MODEL_GEN = "D:\\IIT BHU\\Research\\CWNU\\Joint-SRVDNet\\model_saves\\Generator"
MODEL_DET = "D:\\IIT BHU\\Research\\CWNU\\Joint-SRVDNet\\model_saves\\Detector"
TRAINING_MODE = "DISJOINT"
MODE = "train"

ALPHA = 2 * 1e-6
BETA = 1e-3
GAMMA = 1e-3
IMG_SIZE = 128

# Individual Training of Super Resolution Network and Vehicle Detection Network
INDIVIDUAL_TRAINING_EPOCHS = 10
SR_LEARNING_RATE = 1e-4
SR_MOMENTUM = 0.9
SR_BATCH_SIZE = 4
SR_DECAY_RATE = 0.1
SR_EPOCHS_PER_DECAY = 5
DET_LEARNING_RATE_1 = 1e-4
DET_LEARNING_RATE_2 = 1e-6
DET_BATCH_SIZE = 16
NMS_THRESHOLD = 0.5
NUM_WORKERS = 4
EPOCHS = 10

# Joint Training
JT_LEARNING_RATE = 1e-4
JT_DECAY_RATE = 0.9991
JT_EPOCHS = 50

# Testing
ORG_DIR = ""
TARGET_DIR = ""
TEST_BATCH_SIZE = 4