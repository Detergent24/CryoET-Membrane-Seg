import torch

#Data paths
TOMO_PATH = 'data/tomo1.mrc'
MASK_PATH = 'data/mask1.mrc'

#Saving directories
SAVE = "output"

#Default vals
K_SLICES = 7
#PATCH_SIZE = 256 #use something div 16 and divides X,Y dim of input tomo
PATCH_SIZE = 128
P_MEM = 0.8

#Model
BASE_CHANNELS = 32

'''
#Training
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 0.0
TRAIN_LENGTH = 50000 #Num patches per epoch
VAL_LENGTH = 5000 #Num patches per epoch
BCE_WEIGHT = 0.5
'''

#Training recommended if training locally
BATCH_SIZE = 2
NUM_WORKERS = 0
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
TRAIN_LENGTH = 2000 #Num patches per epoch
VAL_LENGTH = 400 #Num patches per epoch
BCE_WEIGHT = 0.5

#Misc.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
