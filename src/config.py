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

num_epochs = 1
batch_size = 500
learning_rate = 5 * 10**-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.warning("using device %s", device)
