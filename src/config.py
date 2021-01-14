# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:33:21 2021

@author: martin
"""

import logging
import torch

seq_len = 1200
input_size = 12
bert_seq_len = 512
bert_input_size = 768
dropout_prob = .5

num_epochs = 10
batch_size = 500
learning_rate = 5 * 10**-4

from_drive = False
def set_from_drive(value = True):
    global from_drive
    from_drive = value
    if from_drive:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        logging.warning("gdrive set as source of models")
if from_drive:
    logging.warning("gdrive set as source of models")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.warning("using device %s", device)

bert_top_k = 0