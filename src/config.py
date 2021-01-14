# -*- coding: utf-8 -*-
"""
Holds global configuration for the framework.

@author: Martin Benes
"""

import logging
import torch

# dimensions
seq_len = 1200
input_size = 12
bert_seq_len = 512
bert_input_size = 768

# training parameters
batch_size = 500
dropout_prob = .5
num_epochs = 10
learning_rate = 5 * 10**-4

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.warning("using device %s", device)

# data source
from_drive = False
def set_from_drive(value = True):
    """Sets to read data and models from drive.
    
    Args:
        values (bool): True if from drive, False if from the local directory.
    """
    global from_drive
    from_drive = value
    if from_drive:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        logging.warning("gdrive set as source of models")
if from_drive:
    logging.warning("gdrive set as source of models")
