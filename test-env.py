import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import spatial
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

from data import load_data_from_pickle, save_test_data, save_train_data
from models import VGG, ResNet18
from train_models import simple_test_batch
from utils import (get_minibatches_idx, get_weighted_minibatches_idx,
                   set_random_seed)

print('import ok')