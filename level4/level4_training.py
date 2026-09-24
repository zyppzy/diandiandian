import argparse
import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from level4_model import UNet
from level4_losses import CombinedLoss, ssim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))#level4文件夹
PROJECT_ROOT = os.path.dirname(BASE_DIR)#项目文件夹
DATA_INPUT=os.path.join(PROJECT_ROOT,"level4_dataset/input")
DATA_OUTPUT=os.path.join(PROJECT_ROOT,"level4_dataset/output")
