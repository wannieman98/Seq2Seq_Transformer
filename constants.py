import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

NUM_EPOCH = 15
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

